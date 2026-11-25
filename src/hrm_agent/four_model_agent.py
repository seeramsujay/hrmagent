"""Supervisor -> Coder -> Verifier pipeline (historical name: "4M").

The former first tier claimed to be an HRM that discovered algorithmic invariants from text. It was
an untrained network plus keyword matching, and has been removed: Sapient's HRM is a grid solver
(see `hrm_agent.hrm_model`) and has no role in text-to-code synthesis.

Every line of code produced here comes from the configured LLM engines. With the default
MockLLMEngine the output is canned and says nothing about solving ability.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from pydantic import ValidationError

from hrm_agent.code_agent import CodeActionAgent
from hrm_agent.llm_engine import LLMEngine, MockLLMEngine
from hrm_agent.patcher import apply_search_replace_patch
from hrm_agent.schemas import ExecutionTelemetry, PlannerBlueprint, ProblemSpec
from judge.sandbox import SandboxRunner


@dataclass
class FourModelTrace:
    """Trace capturing the complete 4-tier reasoning and code synthesis pipeline."""

    problem_id: str
    solved: bool = False
    final_verdict: str = "UNKNOWN"
    blueprint: Optional[PlannerBlueprint] = None
    synthesized_code: str = ""
    verification_telemetry: Optional[ExecutionTelemetry] = None
    repair_iterations: int = 0
    patches_applied: list[str] = field(default_factory=list)
    language: str = "python"


class FourModelAgent:
    """Supervisor planner, code worker, and sandbox-verified repair."""

    def __init__(
        self,
        supervisor_engine: Optional[LLMEngine] = None,
        coder_engine: Optional[LLMEngine] = None,
        verifier_engine: Optional[LLMEngine] = None,
        sandbox: Optional[SandboxRunner] = None,
        workspace_dir: Optional[Path] = None,
    ):
        self.supervisor = supervisor_engine or MockLLMEngine()
        self.coder = coder_engine or MockLLMEngine()
        self.verifier = verifier_engine or MockLLMEngine()
        self.sandbox = sandbox or SandboxRunner()
        self.code_agent = CodeActionAgent(workspace=workspace_dir)

    def solve(
        self,
        spec: ProblemSpec,
        language: Literal["python", "cpp", "typescript", "kotlin"] = "python",
        max_repair_iters: int = 2,
    ) -> FourModelTrace:
        """Plan, write code, then verify against the samples and repair."""
        trace = FourModelTrace(problem_id=spec.id, language=language)
        if len(spec.sample_inputs) != len(spec.sample_outputs):
            # zip() would silently skip the unmatched samples and could report AC for them.
            trace.final_verdict = "INVALID_SAMPLES"
            return trace
        task = (
            f"{spec.title} - {spec.description}\nInput format: {spec.input_format}\n"
            f"Output format: {spec.output_format}\nConstraints: {spec.constraints}"
        )

        bp_raw, _ = self.supervisor.generate(
            f"System 2 algorithmic planner.\nProblem: {task}\n"
            "Emit a JSON blueprint with time complexity, data structures, and a step checklist.",
            max_tokens=300,
        )
        try:
            trace.blueprint = PlannerBlueprint.model_validate_json(_strip_fences(bp_raw))
        except ValidationError:
            trace.blueprint = None

        code_raw, _ = self.coder.generate(
            f"# Language: {language}\n# Problem: {task}\n# Plan: {bp_raw}\n"
            "# Emit a complete executable program that reads stdin and writes stdout.",
            max_tokens=1024,
        )
        code = _strip_fences(code_raw)
        trace.synthesized_code = code

        # The sandbox only runs Python and C++.
        if language not in ("python", "cpp"):
            trace.final_verdict = "UNSUPPORTED_LANGUAGE"
            return trace

        patch_error = ""
        for iter_idx in range(max_repair_iters + 1):
            telem: Optional[ExecutionTelemetry] = None
            failed_input = failed_expected = ""
            for inp, expected in zip(spec.sample_inputs, spec.sample_outputs):
                telem = self.sandbox.run_code(
                    code=code,
                    language=language,
                    input_data=inp,
                    expected_output=expected,
                    time_limit_ms=spec.time_limit_ms,
                    memory_limit_mb=spec.memory_limit_mb,
                    custom_checker=spec.checker_code,
                )
                if telem.verdict != "AC":
                    failed_input, failed_expected = inp, expected
                    break

            trace.verification_telemetry = telem
            if telem is None:  # no samples: nothing was verified
                break
            trace.final_verdict = telem.verdict
            if telem.verdict == "AC":
                trace.solved = True
                break
            if iter_idx == max_repair_iters:
                break

            trace.repair_iterations += 1
            patch, _ = self.verifier.generate(
                f"Task:\n{task}\nThe program failed with {telem.verdict}.\nInput:\n{failed_input}\n"
                f"Expected output:\n{failed_expected}\n"
                f"Stdout:\n{telem.stdout[:500]}\nStderr:\n{telem.stderr[:500]}\nProgram:\n{code}\n"
                f"{patch_error}"
                "Reply with one block in exactly this form:\n"
                "<<<<<<< SEARCH\nold lines\n=======\nnew lines\n>>>>>>> REPLACE",
                max_tokens=600,
            )
            # The patcher calls Python "py".
            ok, patched, meta = apply_search_replace_patch(
                code, patch, language="py" if language == "python" else language
            )
            patch_error = "" if ok else f"Your previous patch was rejected: {meta.validation_error}\n"
            if ok:
                trace.patches_applied.append(patch)
                code = patched
                trace.synthesized_code = code

        return trace


def _strip_fences(text: str) -> str:
    """Return the body of the first fenced block, or the text unchanged.

    Fences must sit on their own lines, so backticks inside source strings are left alone.
    """
    # As in CommonMark, the closing fence may be longer than the opening one.
    m = re.search(r"^[ \t]*(`{3,})[\w+-]*[ \t]*\n(.*?)^[ \t]*\1`*[ \t]*$", text, re.DOTALL | re.MULTILINE)
    return m.group(2) if m else text
