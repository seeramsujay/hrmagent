"""Model Context Protocol (MCP) Server for hrmagent (HRM-4M).

Exposes HRM latent invariant discovery, 4-Model solving, 3-node closed-loop
solving, sandboxed Code-Action execution, wait4 verification, and TerminalBench
evaluation over stdio or HTTP/SSE for AGY, Claude Code, Claude Desktop, and Zed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Literal, Optional

from mcp.server.mcpserver import MCPServer

from hrm_agent.code_agent import CodeActionAgent
from hrm_agent.four_model_agent import FourModelAgent
from hrm_agent.hrm_latent_engine import HRMLatentSolver
from hrm_agent.hrm_loop import HRMAgentLoop
from hrm_agent.llm_engine import MockLLMEngine
from hrm_agent.schemas import ProblemSpec
from judge.sandbox import SandboxRunner
from judge.terminal_bench import TerminalBenchEvaluator

logger = logging.getLogger("hrmagent.mcp")

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAMMAR_DIR = REPO_ROOT / "grammars"


class HrmAgentMcpApp:
    """MCP server exposing HRM-4M reasoning, verification, and TerminalBench tools."""

    def __init__(self, mode: str = "mock", device: str = "auto"):
        self.mode = mode
        self.device = device
        self.latent_solver = HRMLatentSolver(device=device)
        self.sandbox = SandboxRunner()

        self.mcp = MCPServer(
            name="hrmagent",
            version="0.1.0",
            instructions=(
                "hrmagent (HRM-4M) provides hierarchical latent reasoning, 4-model "
                "algorithmic synthesis, sandboxed Python Code-Action execution, "
                "os.wait4 telemetry verification, and TerminalBench diagnostics."
            ),
        )
        self._register_tools()
        self._register_resources()

    def _register_tools(self) -> None:
        @self.mcp.tool()
        def hrm_discover_invariant(problem_description: str, constraints: str = "") -> str:
            """Discover mathematical/algorithmic invariant in latent space via Sapient-style HRM recurrence."""
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    res = self.latent_solver.discover_invariant(problem_description, constraints)
                return json.dumps(res, indent=2)
            except Exception as exc:
                return json.dumps({"status": "error", "error": str(exc)}, indent=2)

        @self.mcp.tool()
        def hrm_solve_4m(
            problem_id: str,
            description: str,
            title: str = "",
            constraints: str = "",
            sample_inputs: Optional[list[str]] = None,
            sample_outputs: Optional[list[str]] = None,
            language: str = "python",
            time_limit_ms: int = 2000,
            memory_limit_mb: int = 256,
        ) -> str:
            """Solve a problem using the 4-Model HRM pipeline (Latent -> Supervisor -> Coder -> Jev Verifier)."""
            try:
                spec = ProblemSpec(
                    id=problem_id,
                    title=title,
                    description=description,
                    constraints=constraints,
                    sample_inputs=sample_inputs or [],
                    sample_outputs=sample_outputs or [],
                    time_limit_ms=time_limit_ms,
                    memory_limit_mb=memory_limit_mb,
                )
                lang: Literal["python", "cpp", "typescript", "kotlin"] = (
                    language if language in ("python", "cpp", "typescript", "kotlin") else "python"  # type: ignore[assignment]
                )
                with contextlib.redirect_stdout(sys.stderr):
                    agent = FourModelAgent(
                        hrm_solver=self.latent_solver,
                        sandbox=self.sandbox,
                        device=self.device,
                    )
                    trace = agent.solve(spec, language=lang)
                return json.dumps(
                    {
                        "problem_id": trace.problem_id,
                        "solved": trace.solved,
                        "final_verdict": trace.final_verdict,
                        "latent_invariant": trace.latent_invariant,
                        "blueprint": trace.blueprint.model_dump() if trace.blueprint else None,
                        "synthesized_code": trace.synthesized_code,
                        "verification_telemetry": (
                            trace.verification_telemetry.model_dump() if trace.verification_telemetry else None
                        ),
                        "repair_iterations": trace.repair_iterations,
                        "patches_applied": trace.patches_applied,
                        "language": trace.language,
                    },
                    indent=2,
                )
            except Exception as exc:
                return json.dumps({"status": "error", "error": str(exc)}, indent=2)

        @self.mcp.tool()
        def hrm_solve_loop(
            problem_id: str,
            description: str,
            title: str = "",
            constraints: str = "",
            sample_inputs: Optional[list[str]] = None,
            sample_outputs: Optional[list[str]] = None,
            language: str = "python",
            mode: str = "mock",
        ) -> str:
            """Solve a problem using the 3-node HRM closed-loop pipeline (Planner -> Coder/Patcher -> Critic)."""
            try:
                spec = ProblemSpec(
                    id=problem_id,
                    title=title,
                    description=description,
                    constraints=constraints,
                    sample_inputs=sample_inputs or [],
                    sample_outputs=sample_outputs or [],
                )
                engine = MockLLMEngine()
                with contextlib.redirect_stdout(sys.stderr):
                    loop = HRMAgentLoop(
                        node1_planner=engine,
                        node2_coder=engine,
                        node3_critic=engine,
                        sandbox=self.sandbox,
                        language=language,
                        grammar_dir=str(GRAMMAR_DIR),
                    )
                    trace = loop.solve(spec)
                return trace.model_dump_json(indent=2)
            except Exception as exc:
                return json.dumps({"status": "error", "error": str(exc)}, indent=2)

        @self.mcp.tool()
        def hrm_run_code_action(python_code: str, workspace_dir: str = ".") -> str:
            """Execute Python code-action with atomic workspace-confined tools (read_file, write_file, run_command, apply_diff)."""
            try:
                ws = Path(workspace_dir).resolve()
                with contextlib.redirect_stdout(sys.stderr):
                    agent = CodeActionAgent(workspace=ws)
                    res = agent.execute_code_action(python_code)
                return json.dumps(
                    {
                        "success": res.success,
                        "output": res.output,
                        "error": res.error,
                        "workspace": str(ws),
                    },
                    indent=2,
                )
            except Exception as exc:
                return json.dumps({"status": "error", "error": str(exc)}, indent=2)

        @self.mcp.tool()
        def hrm_verify_code(
            code: str,
            language: str = "python",
            sample_inputs: Optional[list[str]] = None,
            sample_outputs: Optional[list[str]] = None,
            time_limit_ms: int = 2000,
            memory_limit_mb: int = 256,
        ) -> str:
            """Verify Python or C++20 code against sample inputs/outputs using SandboxRunner with os.wait4 telemetry."""
            try:
                inputs = sample_inputs or [""]
                outputs = sample_outputs or [""]
                lang: Literal["cpp", "python"] = "cpp" if language == "cpp" else "python"
                results = []
                all_ac = True
                with contextlib.redirect_stdout(sys.stderr):
                    for idx, (inp, exp) in enumerate(zip(inputs, outputs)):
                        telem = self.sandbox.run_code(
                            code=code,
                            language=lang,
                            input_data=inp,
                            expected_output=exp,
                            time_limit_ms=time_limit_ms,
                            memory_limit_mb=memory_limit_mb,
                        )
                        if telem.verdict != "AC":
                            all_ac = False
                        results.append({"case_index": idx, **telem.model_dump()})
                return json.dumps({"all_passed": all_ac, "cases": results}, indent=2)
            except Exception as exc:
                return json.dumps({"status": "error", "error": str(exc)}, indent=2)

        @self.mcp.tool()
        def hrm_run_terminalbench() -> str:
            """Execute the 5-task TerminalBench diagnostic and execution benchmark suite."""
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    evaluator = TerminalBenchEvaluator()
                    summary = evaluator.evaluate_all()
                return json.dumps(summary, indent=2)
            except Exception as exc:
                return json.dumps({"status": "error", "error": str(exc)}, indent=2)

        @self.mcp.tool()
        def hrm_get_status() -> str:
            """Return hrmagent service version, mode, device, CUDA availability, and available tools."""
            cuda_avail = False
            gpu_name = None
            try:
                import torch

                cuda_avail = torch.cuda.is_available()
                if cuda_avail:
                    gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                pass
            return json.dumps(
                {
                    "service": "hrmagent",
                    "version": "0.1.0",
                    "mode": self.mode,
                    "device": self.device,
                    "cuda_available": cuda_avail,
                    "gpu_name": gpu_name,
                    "sandbox_engine": self.sandbox.engine,
                    "grammar_dir": str(GRAMMAR_DIR),
                },
                indent=2,
            )

    def _register_resources(self) -> None:
        @self.mcp.resource("hrm://status")
        def status_resource() -> str:
            """Current hrmagent status and environment capabilities."""
            return json.dumps(
                {
                    "service": "hrmagent",
                    "version": "0.1.0",
                    "mode": self.mode,
                    "device": self.device,
                    "sandbox_engine": self.sandbox.engine,
                },
                indent=2,
            )

        @self.mcp.resource("hrm://grammars")
        def grammars_resource() -> str:
            """List of available GBNF grammar files."""
            files = sorted(p.name for p in GRAMMAR_DIR.glob("*.gbnf")) if GRAMMAR_DIR.exists() else []
            return json.dumps({"grammar_dir": str(GRAMMAR_DIR), "grammars": files}, indent=2)

    def run(self, transport: str = "stdio", **kwargs: Any) -> None:
        self.mcp.run(transport=transport, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="hrmagent Model Context Protocol (MCP) Server")
    parser.add_argument("--transport", choices=["stdio", "streamable-http", "sse"], default="stdio")
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--mode", choices=["mock", "dual-pair"], default="mock")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    app = HrmAgentMcpApp(mode=args.mode, device=args.device)
    if args.transport == "stdio":
        app.run(transport="stdio")
    else:
        app.run(transport=args.transport, port=args.port)


if __name__ == "__main__":
    main()
