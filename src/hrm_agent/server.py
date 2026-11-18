"""FastAPI Localhost Server for HRM Competitive Programming Dual-Pair Agent.

Exposes:
- Standard OpenAI API endpoints (/v1/chat/completions, /v1/completions, /v1/models)
  allowing benchmarking and evaluation harnesses (Aider, SWE-bench, lm-eval, etc.) to
  plug in natively.
- Dedicated HRM closed-loop endpoints (/solve, /evaluate) for competitive programming
  problem specifications and multi-node execution traces.
- Health, GPU telemetry, and model status (/health, /).
"""

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from hrm_agent.hrm_loop import HRMAgentLoop
from hrm_agent.llm_engine import (
    DualPairEngine,
    LLMEngine,
    MockLLMEngine,
)
from hrm_agent.schemas import AgentTrace, ProblemSpec
from judge.sandbox import SandboxRunner

# =====================================================================
# OpenAI Request / Response Schemas
# =====================================================================


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: Any = ""
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str = "hrm-dual-pair"
    messages: list[ChatMessage]
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: Optional[int] = 1024
    stream: bool = False
    stop: Optional[list[str] | str] = None
    n: int = 1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    user: Optional[str] = None


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: UsageInfo


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    prompt: str
    model: str = "hrm-dual-pair"
    max_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 1731974400
    owned_by: str = "hrm-agent"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelCard]


class EvaluateRequest(BaseModel):
    problems: list[ProblemSpec]
    language: Literal["cpp", "python"] = "python"


class EvaluateResponse(BaseModel):
    total_problems: int
    solved_count: int
    pass_at_1: float
    results: list[dict[str, Any]]


# =====================================================================
# Server State & Application Factory
# =====================================================================


class ServerState:
    def __init__(
        self,
        mode: str = "dual-pair",
        role1_path: str = "outputs/merged/role1_architect_critic",
        role2_path: str = "outputs/merged/role2_coder_patcher",
        base_model: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        role1_adapter: str | None = "outputs/checkpoints_architect_critic/final_adapter",
        role2_adapter: str | None = "outputs/checkpoints_dpo/final_dpo_adapter",
        device: str = "auto",
        api_key: str | None = None,
    ):
        self.mode = mode
        self.device = device
        self.api_key = api_key or os.environ.get("HRM_API_KEY")
        self.dual_pair: DualPairEngine | None = None
        self.mock_engine: MockLLMEngine | None = None
        self.sandbox = SandboxRunner()

        if mode == "mock":
            self.mock_engine = MockLLMEngine()
        else:
            try:
                self.dual_pair = DualPairEngine(
                    role1_path=role1_path,
                    role2_path=role2_path,
                    base_model=base_model,
                    role1_adapter=role1_adapter,
                    role2_adapter=role2_adapter,
                    device=device,
                    mock_fallback=False,
                )
            except Exception as e:
                # Strictly fail startup if real model fails to load without explicit mock mode
                raise RuntimeError(
                    f"Failed to load DualPairEngine (mode={mode}, role1={role1_path}, role2={role2_path}): {e}. "
                    "Aborting startup. Pass mode='mock' explicitly if mock inference is intended."
                ) from e

    def verify_api_key(self, authorization: str | None) -> bool:
        """Verify API key from Authorization header if a key is enforced."""
        if not self.api_key:
            return True  # Open localhost access for standard harnesses
        if not authorization:
            return False
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1] == self.api_key
        return authorization == self.api_key

    def get_engines(self) -> tuple[LLMEngine, LLMEngine, LLMEngine]:
        """Return (planner, coder, critic) engines."""
        if self.mock_engine is not None:
            return self.mock_engine, self.mock_engine, self.mock_engine
        if self.dual_pair is not None:
            return (
                self.dual_pair.get_planner_engine(),
                self.dual_pair.get_coder_engine(),
                self.dual_pair.get_critic_engine(),
            )
        raise RuntimeError("No inference engine initialized.")


def create_app(
    mode: str = "dual-pair",
    device: str = "auto",
    role1_path: str = "outputs/merged/role1_architect_critic",
    role2_path: str = "outputs/merged/role2_coder_patcher",
    api_key: str | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        if not hasattr(app.state, "server_state") or app.state.server_state is None:
            app.state.server_state = ServerState(
                mode=mode,
                role1_path=role1_path,
                role2_path=role2_path,
                device=device,
                api_key=api_key,
            )
        yield

    app = FastAPI(
        title="HRM Competitive Programming Agent Server",
        description="Local-First Dual-Pair HRM Agent API with OpenAI harness compatibility",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_state() -> ServerState:
        if not hasattr(app.state, "server_state") or app.state.server_state is None:
            app.state.server_state = ServerState(
                mode=mode,
                role1_path=role1_path,
                role2_path=role2_path,
                device=device,
                api_key=api_key,
            )
        return app.state.server_state

    def _validate_spec_checker(spec: ProblemSpec) -> None:
        """Reject dangerous untrusted checker code at entry boundary."""
        if spec.checker_code:
            forbidden = ["os.system", "subprocess", "socket", "shutil.rmtree", "__import__('os')", "open("]
            if any(tok in spec.checker_code for tok in forbidden):
                raise HTTPException(
                    status_code=400,
                    detail="Security violation: dangerous operations in checker_code are forbidden.",
                )

    @app.get("/")
    def root():
        state = get_state()
        return {
            "service": "hrm-agent-server",
            "version": "0.1.0",
            "status": "ready",
            "mode": state.mode,
            "openapi": "/openapi.json",
            "docs": "/docs",
            "api_key_status": (
                "enforced" if state.api_key else "open_localhost (any key accepted, e.g. sk-hrm-localhost)"
            ),
            "endpoints": [
                "/v1/chat/completions",
                "/v1/completions",
                "/v1/models",
                "/v1/api_key",
                "/solve",
                "/evaluate",
                "/health",
            ],
        }

    @app.get("/v1/api_key")
    def get_api_key():
        state = get_state()
        return {
            "api_key_required": state.api_key is not None,
            "default_key": state.api_key or "sk-hrm-localhost",
            "header": "Authorization: Bearer <key>",
            "openapi_url": "/openapi.json",
            "docs_url": "/docs",
            "models": ["hrm-dual-pair", "hrm-architect-critic", "hrm-coder-patcher"],
            "base_url": "http://localhost:8000/v1",
        }

    @app.get("/health")
    def health():
        state = get_state()
        cuda_avail = False
        gpu_name = None
        vram_allocated_mb = 0.0
        vram_reserved_mb = 0.0

        try:
            import torch

            cuda_avail = torch.cuda.is_available()
            if cuda_avail:
                gpu_name = torch.cuda.get_device_name(0)
                vram_allocated_mb = round(torch.cuda.memory_allocated() / (1024**2), 2)
                vram_reserved_mb = round(torch.cuda.memory_reserved() / (1024**2), 2)
        except Exception:
            pass

        return {
            "status": "healthy",
            "service": "hrm-agent",
            "mode": state.mode,
            "device": state.device,
            "cuda_available": cuda_avail,
            "gpu_name": gpu_name,
            "vram_allocated_mb": vram_allocated_mb,
            "vram_reserved_mb": vram_reserved_mb,
            "models": [
                "hrm-dual-pair",
                "hrm-architect-critic",
                "hrm-coder-patcher",
            ],
        }

    @app.get("/v1/models", response_model=ModelListResponse)
    def list_models():
        models = [
            ModelCard(id="hrm-dual-pair"),
            ModelCard(id="hrm-architect-critic"),
            ModelCard(id="hrm-coder-patcher"),
        ]
        return ModelListResponse(data=models)

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest, request: Request):
        state = get_state()
        auth = request.headers.get("Authorization")
        if not state.verify_api_key(auth):
            raise HTTPException(status_code=401, detail="Unauthorized: Invalid API key")

        planner, coder, critic = state.get_engines()

        # Serialize complete ordered dialogue history
        system_content = "You are an algorithmic architect and critic."
        dialogue_turns = []
        user_queries = []
        for m in req.messages:
            msg_text = ""
            if isinstance(m.content, str):
                msg_text = m.content
            elif isinstance(m.content, list):
                parts = [str(item.get("text", "")) for item in m.content if isinstance(item, dict) and "text" in item]
                msg_text = "\n".join(parts) if parts else str(m.content)
            elif m.content is not None:
                msg_text = str(m.content)

            if m.role == "system":
                system_content = msg_text
            else:
                dialogue_turns.append(f"<|im_start|>{m.role}\n{msg_text}\n<|im_end|>")
                if m.role == "user":
                    user_queries.append(msg_text)

        full_history = "\n".join(dialogue_turns)
        user_content = (
            "\n\n".join(user_queries) if user_queries else (str(req.messages[-1].content) if req.messages else "")
        )

        # Model routing
        model_name = req.model.lower()
        if "architect" in model_name or "critic" in model_name:
            prompt = f"<|im_start|>system\n{system_content}\n<|im_end|>\n{full_history}\n<|im_start|>assistant\n"
            text, telem = planner.generate(
                prompt=prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            )
        elif "coder" in model_name or "patcher" in model_name:
            prompt = (
                f"<|im_start|>system\n{system_content or 'You are a fast code synthesis engine.'}\n<|im_end|>\n"
                f"{full_history}\n"
                "<|im_start|>assistant\n"
            )
            text, telem = coder.generate(
                prompt=prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            )
        else:
            # Full Dual-Pair HRM synthesis: Architect generates blueprint, Coder writes code
            bp_prompt = (
                "<|im_start|>system\n"
                "You are an expert competitive programming System 2 algorithmic planner. "
                "Output an algorithmic blueprint strictly adhering to the JSON schema. "
                "Keep the blueprint dense, precise, and under 250 tokens. Never write full source code.\n"
                "<|im_end|>\n"
                f"{full_history}\n"
                "<|im_start|>assistant\n"
            )
            bp_text, bp_telem = planner.generate(prompt=bp_prompt, max_tokens=300, temperature=0.2)

            coder_prompt = (
                "<|im_start|>system\n"
                "You are a System 1 fast coder translating algorithmic blueprints directly into optimal code. "
                "Emit only clean, complete code.\n"
                "<|im_end|>\n"
                f"<|im_start|>user\nProblem:\n{user_content}\n\nBlueprint:\n{bp_text}\n<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            text, coder_telem = coder.generate(
                prompt=coder_prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            )
            telem = coder_telem
            telem.prompt_tokens += bp_telem.prompt_tokens
            telem.completion_tokens += bp_telem.completion_tokens

        if req.stream:
            # Return SSE stream compliant with OpenAI API
            def sse_generator():
                chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                created_ts = int(time.time())
                chunk_data = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse_generator(), media_type="text/event-stream")

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=req.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=telem.prompt_tokens,
                completion_tokens=telem.completion_tokens,
                total_tokens=telem.prompt_tokens + telem.completion_tokens,
            ),
        )

    @app.post("/v1/completions", response_model=CompletionResponse)
    def completions(req: CompletionRequest, request: Request):
        state = get_state()
        auth = request.headers.get("Authorization")
        if not state.verify_api_key(auth):
            raise HTTPException(status_code=401, detail="Unauthorized: Invalid API key")

        planner, coder, critic = state.get_engines()
        chosen = coder if "coder" in req.model.lower() else planner
        text, telem = chosen.generate(
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=req.model,
            choices=[CompletionChoice(index=0, text=text, finish_reason="stop")],
            usage=UsageInfo(
                prompt_tokens=telem.prompt_tokens,
                completion_tokens=telem.completion_tokens,
                total_tokens=telem.prompt_tokens + telem.completion_tokens,
            ),
        )

    @app.post("/solve", response_model=AgentTrace)
    @app.post("/v1/solve", response_model=AgentTrace)
    def solve(spec: ProblemSpec, request: Request, language: Literal["cpp", "python"] = "python"):
        _validate_spec_checker(spec)
        state = get_state()
        auth = request.headers.get("Authorization")
        if not state.verify_api_key(auth):
            raise HTTPException(status_code=401, detail="Unauthorized: Invalid API key")

        planner, coder, critic = state.get_engines()
        loop = HRMAgentLoop(
            node1_planner=planner,
            node2_coder=coder,
            node3_critic=critic,
            sandbox=state.sandbox,
            language=language,
        )
        try:
            trace = loop.solve(spec)
            return trace
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"HRM loop solve failed: {e}")

    @app.post("/evaluate", response_model=EvaluateResponse)
    @app.post("/v1/evaluate", response_model=EvaluateResponse)
    def evaluate(eval_req: EvaluateRequest, request: Request):
        state = get_state()
        auth = request.headers.get("Authorization")
        if not state.verify_api_key(auth):
            raise HTTPException(status_code=401, detail="Unauthorized: Invalid API key")

        for p in eval_req.problems:
            _validate_spec_checker(p)

        planner, coder, critic = state.get_engines()
        loop = HRMAgentLoop(
            node1_planner=planner,
            node2_coder=coder,
            node3_critic=critic,
            sandbox=state.sandbox,
            language=eval_req.language,
        )

        results = []
        solved_count = 0
        for p in eval_req.problems:
            trace = loop.solve(p)
            if trace.solved:
                solved_count += 1
            results.append(
                {
                    "problem_id": p.id,
                    "solved": trace.solved,
                    "final_verdict": trace.final_verdict,
                    "macro_steps": trace.macro_steps,
                    "micro_iterations": trace.micro_iterations,
                    "critic_iterations": trace.critic_iterations,
                    "final_code": trace.final_code,
                }
            )

        total = len(eval_req.problems)
        pass_at_1 = round(solved_count / total, 4) if total > 0 else 0.0

        return EvaluateResponse(
            total_problems=total,
            solved_count=solved_count,
            pass_at_1=pass_at_1,
            results=results,
        )

    return app


app = create_app(mode="mock")
