"""Agent Client Protocol (ACP) stdio JSON-RPC 2.0 Server for hrmagent (HRM-4M).

Implements ACP v1 (`initialize`, `session/new`, `session/set_mode`, `session/prompt`,
`session/cancel`) over newline-delimited JSON-RPC on stdin/stdout, enabling
Antigravity (`agy`), Claude Code, Zed, and remote-zed to invoke HRM-4M reasoning,
code-action sandboxing, and TerminalBench verification natively.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import re
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TextIO

from hrm_agent.acp_client import AcpLLMEngine
from hrm_agent.code_agent import CodeActionAgent
from hrm_agent.four_model_agent import FourModelAgent
from hrm_agent.hrm_latent_engine import HRMLatentSolver
from hrm_agent.hrm_loop import HRMAgentLoop
from hrm_agent.llm_engine import LLMEngine, MockLLMEngine
from hrm_agent.schemas import ProblemSpec
from judge.sandbox import SandboxRunner
from judge.terminal_bench import TerminalBenchEvaluator

logger = logging.getLogger("hrmagent.acp")

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAMMAR_DIR = REPO_ROOT / "grammars"


@dataclass
class HrmAcpSession:
    """Per-session ACP state."""

    session_id: str
    cwd: str
    mode: str = "4m"
    cancelled: bool = False
    active_task: Optional[asyncio.Task[Any]] = field(default=None, repr=False)


def _extract_prompt_text(prompt_field: Any) -> str:
    if isinstance(prompt_field, str):
        return prompt_field.strip()
    if isinstance(prompt_field, list):
        parts: list[str] = []
        for block in prompt_field:
            if isinstance(block, dict):
                btype = block.get("type", "text")
                if btype == "text":
                    parts.append(str(block.get("text", "")))
                elif btype == "resource" and isinstance(block.get("resource"), dict):
                    res = block["resource"]
                    if "text" in res:
                        parts.append(str(res["text"]))
                elif "text" in block:
                    parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p).strip()
    return str(prompt_field or "").strip()


def _parse_problem_spec(prompt_text: str) -> ProblemSpec:
    """Parse a JSON ProblemSpec if provided, otherwise synthesize one from text."""
    try:
        match = re.search(r"\{[\s\S]*\}", prompt_text)
        if match:
            data = json.loads(match.group(0))
            if isinstance(data, dict) and ("description" in data or "id" in data):
                return ProblemSpec.model_validate(
                    {
                        "id": data.get("id", f"ACP-{uuid.uuid4().hex[:6]}"),
                        "title": data.get("title", "ACP Problem"),
                        "description": data.get("description", prompt_text),
                        **{k: v for k, v in data.items() if k not in ("id", "title", "description")},
                    }
                )
    except Exception:
        pass

    first_line = prompt_text.splitlines()[0][:64] if prompt_text else "Interactive Task"
    return ProblemSpec(
        id=f"ACP-{uuid.uuid4().hex[:6]}",
        title=first_line,
        description=prompt_text,
        constraints="1 <= N <= 2*10^5",
    )


class HrmAgentAcpServer:
    """ACP v1 JSON-RPC 2.0 stdio server for HRM-4M."""

    def __init__(
        self,
        default_mode: str = "4m",
        device: str = "auto",
        acp_backend: Optional[str] = None,
        out_stream: Optional[TextIO] = None,
    ):
        self.default_mode = default_mode if default_mode in ("4m", "dual-pair", "code-action") else "4m"
        self.device = device
        self.acp_backend = acp_backend
        self._out: TextIO = out_stream or sys.stdout
        self._write_lock = asyncio.Lock()

        self.latent_solver = HRMLatentSolver(device=device)
        self.sandbox = SandboxRunner()
        self.sessions: dict[str, HrmAcpSession] = {}

    def _get_llm_engine(self, cwd: str) -> LLMEngine:
        if self.acp_backend:
            return AcpLLMEngine(backend=self.acp_backend, cwd=cwd)
        return MockLLMEngine()

    async def _send_json(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, separators=(",", ":")) + "\n"
        async with self._write_lock:
            self._out.write(line)
            self._out.flush()

    async def _send_result(self, req_id: Any, result: dict[str, Any]) -> None:
        await self._send_json({"jsonrpc": "2.0", "id": req_id, "result": result})

    async def _send_error(self, req_id: Any, code: int, message: str) -> None:
        await self._send_json({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})

    async def _send_update(self, session_id: str, update: dict[str, Any]) -> None:
        await self._send_json(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {"sessionId": session_id, "update": update},
            }
        )

    async def _emit_thought(self, session_id: str, text: str) -> None:
        await self._send_update(
            session_id,
            {"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": text}},
        )

    async def _emit_message(self, session_id: str, text: str) -> None:
        await self._send_update(
            session_id,
            {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": text}},
        )

    async def _emit_tool_start(self, session_id: str, tool_call_id: str, title: str, kind: str = "execute") -> None:
        await self._send_update(
            session_id,
            {
                "sessionUpdate": "tool_call",
                "toolCallId": tool_call_id,
                "title": title,
                "kind": kind,
                "status": "in_progress",
            },
        )

    async def _emit_tool_done(
        self, session_id: str, tool_call_id: str, summary: str, status: str = "completed"
    ) -> None:
        await self._send_update(
            session_id,
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": tool_call_id,
                "status": status,
                "content": [{"type": "content", "content": {"type": "text", "text": summary}}],
            },
        )

    async def handle_message(self, msg: dict[str, Any]) -> Optional[dict[str, Any]]:
        method = msg.get("method")
        req_id = msg.get("id")
        params = msg.get("params") or {}

        if method == "initialize":
            result = {
                "protocolVersion": 1,
                "agentCapabilities": {
                    "loadSession": False,
                    "promptCapabilities": {"image": False, "embeddedContext": True},
                },
                "agentInfo": {
                    "name": "hrmagent",
                    "title": "HRM-4M Coding Agent",
                    "version": "0.1.0",
                },
                "authMethods": [],
            }
            if req_id is not None:
                await self._send_result(req_id, result)
            return result

        if method == "session/new":
            sid = f"hrm_{uuid.uuid4().hex[:8]}"
            cwd = str(params.get("cwd") or Path.cwd())
            self.sessions[sid] = HrmAcpSession(session_id=sid, cwd=cwd, mode=self.default_mode)
            result = {
                "sessionId": sid,
                "modes": {
                    "currentModeId": self.default_mode,
                    "availableModes": [
                        {
                            "id": "4m",
                            "name": "Four-Model HRM",
                            "description": "Latent Reasoner -> Supervisor -> Code-Action Worker -> Jev Verifier",
                        },
                        {
                            "id": "dual-pair",
                            "name": "3-Node HRM Loop",
                            "description": "Planner -> Coder/Patcher -> Jev Critic closed loop",
                        },
                        {
                            "id": "code-action",
                            "name": "Code-Action Sandbox",
                            "description": "Execute atomic Python code-actions confined to session cwd",
                        },
                    ],
                },
            }
            if req_id is not None:
                await self._send_result(req_id, result)
            return result

        if method == "session/set_mode":
            sid = params.get("sessionId", "")
            mode_id = params.get("modeId", "")
            session = self.sessions.get(sid)
            if not session:
                if req_id is not None:
                    await self._send_error(req_id, -32602, f"Unknown sessionId: {sid}")
                return None
            if mode_id not in ("4m", "dual-pair", "code-action"):
                if req_id is not None:
                    await self._send_error(req_id, -32602, f"Unsupported modeId: {mode_id}")
                return None
            session.mode = mode_id
            if req_id is not None:
                await self._send_result(req_id, {})
            return {}

        if method == "session/cancel":
            sid = params.get("sessionId", "")
            session = self.sessions.get(sid)
            if session:
                session.cancelled = True
                if session.active_task and not session.active_task.done():
                    session.active_task.cancel()
            return None

        if method == "session/prompt":
            sid = params.get("sessionId", "")
            session = self.sessions.get(sid)
            if not session:
                if req_id is not None:
                    await self._send_error(req_id, -32602, f"Unknown sessionId: {sid}")
                return None

            session.cancelled = False
            prompt_text = _extract_prompt_text(params.get("prompt"))
            task = asyncio.create_task(self._run_prompt_turn(session, prompt_text))
            session.active_task = task
            try:
                stop_reason = await task
            except asyncio.CancelledError:
                stop_reason = "cancelled"
            finally:
                session.active_task = None

            result = {"stopReason": stop_reason}
            if req_id is not None:
                await self._send_result(req_id, result)
            return result

        if req_id is not None:
            await self._send_error(req_id, -32601, f"Method not found: {method}")
        return None

    async def _run_prompt_turn(self, session: HrmAcpSession, prompt_text: str) -> str:
        sid = session.session_id
        if not prompt_text:
            await self._emit_message(
                sid, "Provide an algorithmic problem, JSON `ProblemSpec`, Python code-action, or `/help`."
            )
            return "end_turn"

        if prompt_text.startswith("/"):
            return await self._handle_slash_command(session, prompt_text)

        if session.mode == "code-action":
            return await self._run_code_action(session, prompt_text)

        spec = _parse_problem_spec(prompt_text)

        # Step 1: Discover latent invariant
        await self._emit_thought(sid, f"Running HRM latent recurrence on `{spec.id}`...\n")
        with contextlib.redirect_stdout(sys.stderr):
            invariant = self.latent_solver.discover_invariant(spec.description, spec.constraints)

        if session.mode == "dual-pair":
            return await self._run_dual_pair(session, spec, invariant)
        return await self._run_four_model(session, spec, invariant)

    async def _handle_slash_command(self, session: HrmAcpSession, command_line: str) -> str:
        sid = session.session_id
        parts = command_line.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "/help":
            await self._emit_message(
                sid,
                "**hrmagent ACP Commands**\n"
                "- `/mode <4m|dual-pair|code-action>` — Switch execution pipeline\n"
                "- `/invariant <problem>` — Discover latent mathematical invariant\n"
                "- `/terminalbench` — Run TerminalBench 5-task diagnostic suite\n"
                "- `/status` — Show active engine mode, sandbox, and backend\n",
            )
            return "end_turn"

        if cmd == "/mode":
            if arg in ("4m", "dual-pair", "code-action"):
                session.mode = arg
                await self._emit_message(sid, f"Switched session `{sid}` mode to `{arg}`.")
            else:
                await self._emit_message(sid, f"Current mode: `{session.mode}`. Usage: `/mode <4m|dual-pair|code-action>`.")
            return "end_turn"

        if cmd == "/status":
            status = {
                "service": "hrmagent",
                "session_id": sid,
                "mode": session.mode,
                "device": self.device,
                "sandbox_engine": self.sandbox.engine,
                "acp_backend": self.acp_backend or "mock",
            }
            await self._emit_message(sid, f"```json\n{json.dumps(status, indent=2)}\n```")
            return "end_turn"

        if cmd == "/invariant":
            with contextlib.redirect_stdout(sys.stderr):
                inv = self.latent_solver.discover_invariant(arg or "Two pointer window", "")
            await self._emit_message(sid, f"```json\n{json.dumps(inv, indent=2)}\n```")
            return "end_turn"

        if cmd == "/terminalbench":
            tc_id = f"tc_{uuid.uuid4().hex[:8]}"
            await self._emit_tool_start(sid, tc_id, "Run TerminalBench diagnostic suite", kind="execute")
            with contextlib.redirect_stdout(sys.stderr):
                evaluator = TerminalBenchEvaluator()
                summary = evaluator.evaluate_all()
            await self._emit_tool_done(
                sid,
                tc_id,
                f"TerminalBench: {summary['passed_tasks']}/{summary['total_tasks']} ({summary['pass_rate_pct']}%)",
            )
            await self._emit_message(sid, f"```json\n{json.dumps(summary, indent=2)}\n```")
            return "end_turn"

        await self._emit_message(sid, f"Unknown command `{cmd}`. Send `/help` for available commands.")
        return "end_turn"

    async def _run_code_action(self, session: HrmAcpSession, prompt_text: str) -> str:
        sid = session.session_id
        tc_id = f"tc_{uuid.uuid4().hex[:8]}"
        await self._emit_tool_start(sid, tc_id, f"Execute Code-Action in {session.cwd}", kind="execute")

        agent = CodeActionAgent(workspace=Path(session.cwd))
        code = agent.extract_code_action(prompt_text) or prompt_text
        with contextlib.redirect_stdout(sys.stderr):
            res = agent.execute_code_action(code)

        await self._emit_tool_done(
            sid,
            tc_id,
            "Code-action succeeded." if res.success else f"Error: {res.error}",
            status="completed" if res.success else "failed",
        )
        await self._emit_message(
            sid,
            f"**Code-Action {'Succeeded' if res.success else 'Failed'}**\n\n"
            f"```\n{res.output or res.error or '(no output)'}\n```",
        )
        return "end_turn"

    async def _run_four_model(
        self, session: HrmAcpSession, spec: ProblemSpec, invariant: dict[str, Any]
    ) -> str:
        sid = session.session_id
        tc_id = f"tc_{uuid.uuid4().hex[:8]}"
        await self._emit_tool_start(sid, tc_id, f"Run 4-Model HRM pipeline on {spec.id}", kind="execute")

        engine = self._get_llm_engine(session.cwd)
        with contextlib.redirect_stdout(sys.stderr):
            agent = FourModelAgent(
                hrm_solver=self.latent_solver,
                supervisor_engine=engine,
                coder_engine=engine,
                verifier_engine=engine,
                sandbox=self.sandbox,
                workspace_dir=Path(session.cwd),
                device=self.device,
            )
            trace = agent.solve(spec, language="python")

        await self._emit_tool_done(
            sid,
            tc_id,
            f"Verdict: {trace.final_verdict} (Solved: {trace.solved})",
            status="completed" if trace.solved or not spec.sample_inputs else "failed",
        )
        bp_json = trace.blueprint.model_dump_json(indent=2) if trace.blueprint else "{}"
        await self._emit_message(
            sid,
            f"### HRM-4M Result ({spec.id})\n\n"
            f"- **Paradigm:** `{invariant.get('paradigm')}`\n"
            f"- **Latent Invariant:** {invariant.get('invariant')}\n"
            f"- **Latent Steps:** `{invariant.get('latent_steps')}` (Halt Prob: `{invariant.get('halting_probability')}`)\n"
            f"- **Verdict:** `{trace.final_verdict}` (Solved: `{trace.solved}`)\n\n"
            f"#### Blueprint\n```json\n{bp_json}\n```\n\n"
            f"#### Synthesized Code\n```python\n{trace.synthesized_code}\n```\n",
        )
        return "end_turn"

    async def _run_dual_pair(
        self, session: HrmAcpSession, spec: ProblemSpec, invariant: dict[str, Any]
    ) -> str:
        sid = session.session_id
        tc_id = f"tc_{uuid.uuid4().hex[:8]}"
        await self._emit_tool_start(sid, tc_id, f"Run 3-Node HRM loop on {spec.id}", kind="execute")

        engine = self._get_llm_engine(session.cwd)
        with contextlib.redirect_stdout(sys.stderr):
            loop = HRMAgentLoop(
                node1_planner=engine,
                node2_coder=engine,
                node3_critic=engine,
                sandbox=self.sandbox,
                language="python",
                grammar_dir=str(GRAMMAR_DIR),
            )
            trace = loop.solve(spec)

        await self._emit_tool_done(
            sid,
            tc_id,
            f"Verdict: {trace.final_verdict} (Solved: {trace.solved})",
            status="completed" if trace.solved else "failed",
        )
        await self._emit_message(
            sid,
            f"### HRM Dual-Pair Result ({spec.id})\n\n"
            f"- **Latent Paradigm:** `{invariant.get('paradigm')}`\n"
            f"- **Verdict:** `{trace.final_verdict}` (Solved: `{trace.solved}`)\n"
            f"- **Macro Steps:** `{trace.macro_steps}` | **Micro Repairs:** `{trace.micro_iterations}`\n\n"
            f"```python\n{trace.final_code or ''}\n```\n",
        )
        return "end_turn"

    async def serve_stdio(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                msg = json.loads(stripped)
            except json.JSONDecodeError as exc:
                await self._send_error(None, -32700, f"Parse error: {exc}")
                continue
            await self.handle_message(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="hrmagent Agent Client Protocol (ACP) stdio server")
    parser.add_argument("--mode", default="4m", choices=["4m", "dual-pair", "code-action"], help="Default ACP mode")
    parser.add_argument("--device", default="auto", help="Compute device ('auto', 'cpu', 'cuda')")
    parser.add_argument(
        "--acp-backend",
        default=None,
        help="Optional external ACP backend ('agy', 'claude', or command) for reasoning nodes",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    server = HrmAgentAcpServer(
        default_mode=args.mode,
        device=args.device,
        acp_backend=args.acp_backend,
    )
    asyncio.run(server.serve_stdio())


if __name__ == "__main__":
    main()
