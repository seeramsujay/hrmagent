"""Hierarchical Reasoning Model (HRM) closed-loop coding orchestrator.

Implements the three-node state machine:
Macro-Step (Node 1 Planner) -> Inner Micro-Loop (Node 2 Coder + Sandbox) -> Jev Gate (Node 3 Critic)
Supports asymmetric test-time scaling, unified-diff patching, and structured JSONL tracing.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from hrm_agent.llm_engine import LLMEngine
from hrm_agent.patcher import apply_patch
from hrm_agent.schemas import (
    AgentTrace,
    ExecutionTelemetry,
    JevVerdict,
    LLMTelemetry,
    PlannerBlueprint,
    ProblemSpec,
)
from judge.fuzz_oracle import DifferentialOracle
from judge.sandbox import SandboxRunner


class HRMAgentLoop:
    """Orchestrates the hierarchical reasoning pipeline for competitive programming."""

    def __init__(
        self,
        node1_planner: LLMEngine,
        node2_coder: LLMEngine,
        node3_critic: LLMEngine,
        sandbox: SandboxRunner | None = None,
        language: str = "cpp",
        n_plans: int = 1,
        n_candidates: int = 3,
        max_inner_iters: int = 3,
        max_critic_iters: int = 2,
        max_planner_escalations: int = 1,
        grammar_dir: str = "grammars",
        trace_file: str | None = None,
    ):
        self.planner = node1_planner
        self.coder = node2_coder
        self.critic = node3_critic
        self.sandbox = sandbox or SandboxRunner()
        self.oracle = DifferentialOracle(self.sandbox)
        self.language = language
        self.n_plans = n_plans
        self.n_candidates = n_candidates
        self.max_inner_iters = max_inner_iters
        self.max_critic_iters = max_critic_iters
        self.max_planner_escalations = max_planner_escalations
        self.grammar_dir = Path(grammar_dir)
        self.trace_file = trace_file

    def _log_trace(self, trace: AgentTrace) -> None:
        """Persist structured execution trace as a JSONL record."""
        if not self.trace_file:
            return
        Path(self.trace_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.trace_file, "a", encoding="utf-8") as f:
            f.write(trace.model_dump_json() + "\n")

    def _run_planner(
        self, spec: ProblemSpec, critique_context: str | None = None
    ) -> tuple[PlannerBlueprint | None, LLMTelemetry]:
        """Node 1: Generate dense <250-token JSON blueprint."""
        prompt = (
            "<|im_start|>system\n"
            "You are an expert competitive programming System 2 algorithmic planner. "
            "Output an algorithmic blueprint strictly adhering to the JSON schema. "
            "Put 'analysis' first so your reasoning precedes constrained slots. "
            "Keep the blueprint dense, precise, and under 250 tokens. Never write full source code.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"Problem ID: {spec.id}\n"
            f"Statement:\n{spec.description}\n"
            f"Input Format:\n{spec.input_format}\n"
            f"Output Format:\n{spec.output_format}\n"
            f"Constraints:\n{spec.constraints}\n"
        )
        if critique_context:
            prompt += f"\nCRITICAL FIX FROM PREVIOUS ATTEMPT:\n{critique_context}\n"
        prompt += "<|im_end|>\n<|im_start|>assistant\n"

        grammar_file = str(self.grammar_dir / "planner.gbnf")
        raw_text, telem = self.planner.generate(
            prompt=prompt,
            grammar_path=grammar_file if os.path.exists(grammar_file) else None,
            max_tokens=300,
            temperature=0.2,
        )

        try:
            data = json.loads(raw_text.strip())
            blueprint = PlannerBlueprint.model_validate(data)
            return blueprint, telem
        except Exception:
            # Fallback heuristic parse if JSON parser throws minor syntax variance
            try:
                start_idx = raw_text.find("{")
                end_idx = raw_text.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    data = json.loads(raw_text[start_idx:end_idx])
                    return PlannerBlueprint.model_validate(data), telem
            except Exception:
                pass
        return None, telem

    def _run_coder_plan_to_code(
        self, spec: ProblemSpec, blueprint: PlannerBlueprint, temp: float = 0.6
    ) -> tuple[str, LLMTelemetry]:
        """Node 2: Plan-to-Code mode generating pure C++20 or Python without markdown fences."""
        lang_str = "C++20" if self.language == "cpp" else "Python 3"
        prompt = (
            "<|im_start|>system\n"
            f"You are a System 1 fast coder translating algorithmic blueprints directly into optimal {lang_str}. "
            "Do NOT include markdown fences, chain-of-thought, or prose explanations. "
            "Emit only clean, compilable, complete code.\n"
        )
        if self.language == "cpp":
            prompt += (
                "Requirements: Begin immediately with #include, use ios::sync_with_stdio(false); cin.tie(nullptr); "
                "use explicit long long aliases when use_int64 is true, and output '\\n'.\n"
            )
        prompt += (
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"Problem: {spec.description}\n"
            f"Constraints: {spec.constraints}\n"
            f"Blueprint: {blueprint.model_dump_json()}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        grammar_file = (
            str(self.grammar_dir / "cpp_raw.gbnf") if self.language == "cpp" else str(self.grammar_dir / "py_raw.gbnf")
        )

        raw_code, telem = self.coder.generate(
            prompt=prompt,
            grammar_path=grammar_file if os.path.exists(grammar_file) else None,
            max_tokens=1024,
            temperature=temp,
        )
        # Strip extraneous markdown backticks if leaked
        clean_code = raw_code.strip()
        if clean_code.startswith("```"):
            clean_code = clean_code.split("\n", 1)[1]
        if clean_code.endswith("```"):
            clean_code = clean_code.rsplit("\n", 1)[0]
        return clean_code.strip(), telem

    def _run_coder_error_to_diff(
        self, spec: ProblemSpec, current_code: str, error_feedback: str
    ) -> tuple[str | None, LLMTelemetry]:
        """Node 2: Error-to-Unified-Diff mode emitting strictly valid unified diffs."""
        prompt = (
            "<|im_start|>system\n"
            "You are a fast code repair patcher. Inspect the code and error feedback, then emit a strictly "
            "valid unified diff (--- a/solution.cpp \\n +++ b/solution.cpp) to fix the bug or bottleneck. "
            "Do not emit markdown backticks or full file rewrites; emit only the unified diff.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"Current Code:\n{current_code}\n\n"
            f"Error / Directive Feedback:\n{error_feedback}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        grammar_file = str(self.grammar_dir / "unified_diff.gbnf")
        diff_text, telem = self.coder.generate(
            prompt=prompt,
            grammar_path=grammar_file if os.path.exists(grammar_file) else None,
            max_tokens=512,
            temperature=0.3,
        )
        return diff_text.strip(), telem

    def _run_critic(
        self, spec: ProblemSpec, current_code: str, stress_telem: ExecutionTelemetry
    ) -> tuple[JevVerdict | None, LLMTelemetry]:
        """Node 3: Jev Critic inspecting code and stress telemetry."""
        prompt = (
            "<|im_start|>system\n"
            "You are Jev, a rigorous competitive programming complexity critic and verifier. "
            "Inspect the code, problem constraints, and execution telemetry. "
            "Emit a JSON verdict adhering strictly to the schema. Put 'analysis' first.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"Problem Constraints: {spec.constraints}\n"
            f"Time Limit: {spec.time_limit_ms} ms, Memory Limit: {spec.memory_limit_mb} MB\n"
            f"Current Solution Code:\n{current_code}\n\n"
            f"Stress Execution Telemetry:\n"
            f"Verdict: {stress_telem.verdict}, Wall Time: {stress_telem.wall_clock_ms} ms, "
            f"CPU Time: {stress_telem.cpu_time_ms} ms, Peak RSS: {stress_telem.peak_rss_kb} KB\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        grammar_file = str(self.grammar_dir / "critic.gbnf")
        raw_text, telem = self.critic.generate(
            prompt=prompt,
            grammar_path=grammar_file if os.path.exists(grammar_file) else None,
            max_tokens=350,
            temperature=0.1,
        )

        try:
            data = json.loads(raw_text.strip())
            return JevVerdict.model_validate(data), telem
        except Exception:
            try:
                start_idx = raw_text.find("{")
                end_idx = raw_text.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    data = json.loads(raw_text[start_idx:end_idx])
                    return JevVerdict.model_validate(data), telem
            except Exception:
                pass
        return None, telem

    def _test_against_samples(self, code: str, spec: ProblemSpec) -> tuple[bool, str | None, ExecutionTelemetry]:
        """Run candidate against sample test cases."""
        last_telem = ExecutionTelemetry(
            verdict="AC",
            wall_clock_ms=0.0,
            cpu_time_ms=0.0,
            peak_rss_kb=0,
            exit_code=0,
            sandbox_engine=self.sandbox.engine,
        )

        for i, (inp, out) in enumerate(zip(spec.sample_inputs, spec.sample_outputs)):
            telem = self.sandbox.run_code(
                code=code,
                language=self.language,
                input_data=inp,
                expected_output=out,
                time_limit_ms=spec.time_limit_ms,
                memory_limit_mb=spec.memory_limit_mb,
                custom_checker=spec.checker_code,
            )
            last_telem = telem
            if telem.verdict != "AC":
                if telem.verdict == "CE":
                    err_msg = f"Compilation Error:\n{telem.stderr}"
                elif telem.verdict == "WA":
                    err_msg = (
                        f"Wrong Answer on Sample {i + 1}:\n"
                        f"Input:\n{inp.strip()}\n"
                        f"Expected:\n{out.strip()}\n"
                        f"Got:\n{telem.stdout.strip()}"
                    )
                else:
                    err_msg = f"{telem.verdict} on Sample {i + 1} (exit {telem.exit_code}):\n{telem.stderr}"
                return False, err_msg, telem

        return True, None, last_telem

    def solve(self, spec: ProblemSpec) -> AgentTrace:
        """Execute the full closed-loop HRM competitive programming solving process."""
        trace = AgentTrace(problem_id=spec.id)
        planner_escalations = 0
        critique_context: str | None = None

        while planner_escalations <= self.max_planner_escalations:
            trace.macro_steps += 1

            # 1. Macro-Step: Algorithmic Planning (Node 1)
            blueprints: list[PlannerBlueprint] = []
            for _ in range(self.n_plans):
                bp, telem = self._run_planner(spec, critique_context)
                trace.llm_metrics.append(telem)
                if bp:
                    blueprints.append(bp)
                    trace.planner_blueprints.append(bp)

            if not blueprints:
                # Fallback generic greedy blueprint if generation failed
                blueprints.append(
                    PlannerBlueprint(
                        analysis="Default linear greedy traversal",
                        paradigm="greedy",
                        data_structures=["vector"],
                        time_complexity="O(N)",
                        space_complexity="O(1)",
                        use_int64=True,
                        edge_cases=["N=1"],
                        steps=["Read input", "Process sequentially", "Output answer"],
                    )
                )

            active_blueprint = blueprints[0]

            # 2. Asymmetric Test-Time Scaling: Draw candidates from Node 2
            candidate_codes: list[str] = []
            for i in range(self.n_candidates):
                temp = 0.6 if i > 0 else 0.2
                code, telem = self._run_coder_plan_to_code(spec, active_blueprint, temp=temp)
                trace.llm_metrics.append(telem)
                candidate_codes.append(code)
                trace.candidate_codes.append(code)

            # Test candidates against samples
            passing_candidate: str | None = None
            last_err_msg: str | None = None
            last_telem: ExecutionTelemetry | None = None

            for code in candidate_codes:
                passed, err_msg, telem = self._test_against_samples(code, spec)
                trace.telemetry_history.append(telem)
                if passed:
                    passing_candidate = code
                    last_telem = telem
                    break
                else:
                    last_err_msg = err_msg
                    last_telem = telem

            # If no candidate passes samples immediately, enter Error-to-Diff micro-loop
            current_code = passing_candidate or candidate_codes[0]
            if not passing_candidate and last_err_msg:
                for inner_iter in range(self.max_inner_iters):
                    trace.micro_iterations += 1
                    diff_text, telem = self._run_coder_error_to_diff(spec, current_code, last_err_msg)
                    trace.llm_metrics.append(telem)
                    trace.patches.append(diff_text)

                    ok, patched_code, _ = apply_patch(current_code, diff_text, language=self.language)
                    if ok:
                        current_code = patched_code
                        passed, err_msg, telem = self._test_against_samples(current_code, spec)
                        trace.telemetry_history.append(telem)
                        if passed:
                            passing_candidate = current_code
                            last_telem = telem
                            break
                        else:
                            last_err_msg = err_msg
                    else:
                        last_err_msg = f"Patch application failed. Please check diff offsets:\n{diff_text}"

            if not passing_candidate:
                # Could not pass sample tests in this macro step
                planner_escalations += 1
                critique_context = f"Candidates failed sample tests with error:\n{last_err_msg}"
                continue

            # 3. Stress Test & Telemetry Gathering
            fuzz_ok, stress_telem = self.oracle.test_candidate_stress(
                code=current_code,
                language=self.language,
                spec=spec,
            )
            trace.telemetry_history.append(stress_telem)

            # 4. Node 3: Jev Critic Gate
            trace.critic_iterations += 1
            verdict, telem = self._run_critic(spec, current_code, stress_telem)
            trace.llm_metrics.append(telem)

            if verdict:
                trace.critic_verdicts.append(verdict)
                if verdict.status == "PASS" and fuzz_ok:
                    trace.solved = True
                    trace.final_verdict = "AC"
                    trace.final_code = current_code
                    break
                elif verdict.escalate_to_planner:
                    planner_escalations += 1
                    critique_context = (
                        f"Critic identified algorithmic flaw: {verdict.analysis}\n"
                        f"Bottleneck class: {verdict.bottleneck_class}\n"
                        f"Patch directive: {verdict.patch_directive}"
                    )
                    continue
                else:
                    # Localized fix routed back to Node 2 via unified diff
                    diff_text, d_telem = self._run_coder_error_to_diff(
                        spec,
                        current_code,
                        f"Critic directive: {verdict.patch_directive}",
                    )
                    trace.llm_metrics.append(d_telem)
                    trace.patches.append(diff_text)
                    ok, patched_code, _ = apply_patch(current_code, diff_text, language=self.language)
                    if ok:
                        s_passed, _, s_telem = self._test_against_samples(patched_code, spec)
                        trace.telemetry_history.append(s_telem)
                        if s_passed:
                            current_code = patched_code
                            trace.solved = True
                            trace.final_verdict = "AC"
                            trace.final_code = current_code
                            break
            else:
                # If critic output failed to parse, accept candidate if samples and fuzz passed
                if fuzz_ok:
                    trace.solved = True
                    trace.final_verdict = "AC"
                    trace.final_code = current_code
                    break

            planner_escalations += 1

        trace.end_time = datetime.now(timezone.utc).isoformat()
        if not trace.final_code and candidate_codes:
            trace.final_code = candidate_codes[0]
            trace.final_verdict = last_telem.verdict if last_telem else "WA"

        self._log_trace(trace)
        return trace
