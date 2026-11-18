"""Property-based & worst-case input generator and differential oracle consensus.

Generates boundary cases (N=1, all-identical, sorted/reverse), worst-case stress
inputs (N=N_max, 10^9 overflow, star/line graphs), and executes multi-candidate
differential oracle consensus.
"""

import random
import re

from hrm_agent.schemas import ExecutionTelemetry, ProblemSpec
from judge.sandbox import SandboxRunner, normalize_output


class FuzzCaseGenerator:
    """Generates property-based, boundary, and stress test cases for CP problems."""

    @staticmethod
    def extract_bounds(constraints: str) -> dict[str, int]:
        """Heuristically extract upper bounds for N and value ranges from constraints text."""
        bounds = {"n_max": 200000, "val_max": 1000000000, "t_max": 100}

        # Look for patterns like 1 <= n <= 2 * 10^5 or n <= 1000
        n_match = re.search(r"[Nn]\s*(?:<=|≤|<)\s*(\d+)(?:\s*\*\s*10\^(\d+))?", constraints)
        if n_match:
            base = int(n_match.group(1))
            power = int(n_match.group(2)) if n_match.group(2) else 0
            if power > 0:
                bounds["n_max"] = min(200000, base * (10**power))
            else:
                bounds["n_max"] = min(200000, base)

        val_match = re.search(r"[Aa]\[[ij]\]\s*(?:<=|≤|<)\s*(\d+)(?:\s*\*\s*10\^(\d+))?", constraints)
        if val_match:
            base = int(val_match.group(1))
            power = int(val_match.group(2)) if val_match.group(2) else 0
            bounds["val_max"] = base * (10**power) if power > 0 else base

        return bounds

    @classmethod
    def generate_boundary_cases(cls, spec: ProblemSpec) -> list[str]:
        """Generate minimum boundary and edge cases (e.g. N=1, zeros, extremes)."""
        cases = []
        cls.extract_bounds(spec.constraints)

        # Case 1: Minimal N=1
        cases.append("1\n1\n0\n")
        cases.append("1\n1\n1000000000\n")

        # Case 2: Minimal array N=2 with identical elements
        cases.append("1\n2\n5 5\n")

        # Case 3: Strictly alternating values
        cases.append("1\n4\n1 2 1 2\n")

        # Case 4: Sorted array
        cases.append("1\n5\n1 2 3 4 5\n")

        # Case 5: Reverse-sorted array
        cases.append("1\n5\n5 4 3 2 1\n")

        # Case 6: All zeros
        cases.append("1\n4\n0 0 0 0\n")

        return cases

    @classmethod
    def generate_worst_case(cls, spec: ProblemSpec) -> str:
        """Generate maximum N stress input designed to trigger TLE or integer overflow."""
        bounds = cls.extract_bounds(spec.constraints)
        n = min(bounds["n_max"], 100000)  # Safe stress size for fast fuzzing
        val_max = bounds["val_max"]

        # Generate large values near 10^9 to force 64-bit overflow if using int32
        lines = ["1", str(n)]
        large_vals = [str(random.randint(val_max // 2, val_max)) for _ in range(n)]
        lines.append(" ".join(large_vals))
        return "\n".join(lines) + "\n"

    @classmethod
    def generate_graph_cases(cls, num_nodes: int = 1000, graph_type: str = "star") -> str:
        """Generate graph edge cases: star, line, or binary tree."""
        lines = ["1", f"{num_nodes} {num_nodes - 1}"]
        if graph_type == "star":
            # Node 1 connected to all other nodes (high degree bottleneck)
            for i in range(2, num_nodes + 1):
                lines.append(f"1 {i}")
        elif graph_type == "line":
            # Deep recursion / stack overflow bottleneck
            for i in range(1, num_nodes):
                lines.append(f"{i} {i + 1}")
        elif graph_type == "binary_tree":
            for i in range(2, num_nodes + 1):
                parent = i // 2
                lines.append(f"{parent} {i}")
        return "\n".join(lines) + "\n"


class DifferentialOracle:
    """Multi-AC differential oracle running consensus across candidates or reference models."""

    def __init__(self, sandbox: SandboxRunner):
        self.sandbox = sandbox

    def test_candidate_stress(
        self,
        code: str,
        language: str,
        spec: ProblemSpec,
        fuzz_cases: list[str] | None = None,
    ) -> tuple[bool, ExecutionTelemetry]:
        """
        Run code against boundary and worst-case stress tests.
        Returns whether it survived without TLE/MLE/RE, along with peak telemetry.
        """
        cases = fuzz_cases or []
        if not cases:
            cases.extend(FuzzCaseGenerator.generate_boundary_cases(spec))
            cases.append(FuzzCaseGenerator.generate_worst_case(spec))

        worst_telemetry = ExecutionTelemetry(
            verdict="AC",
            wall_clock_ms=0.0,
            cpu_time_ms=0.0,
            peak_rss_kb=0,
            exit_code=0,
            stdout="",
            stderr="",
            sandbox_engine=self.sandbox.engine,
        )

        for test_in in cases:
            res = self.sandbox.run_code(
                code=code,
                language="cpp" if language == "cpp" else "python",
                input_data=test_in,
                expected_output=None,
                time_limit_ms=spec.time_limit_ms,
                memory_limit_mb=spec.memory_limit_mb,
            )

            # Accumulate peak resource metrics
            worst_telemetry.cpu_time_ms = max(worst_telemetry.cpu_time_ms, res.cpu_time_ms)
            worst_telemetry.wall_clock_ms = max(worst_telemetry.wall_clock_ms, res.wall_clock_ms)
            worst_telemetry.peak_rss_kb = max(worst_telemetry.peak_rss_kb, res.peak_rss_kb)

            if res.verdict != "AC":
                worst_telemetry.verdict = res.verdict
                worst_telemetry.stderr = res.stderr
                worst_telemetry.exit_code = res.exit_code
                return False, worst_telemetry

        return True, worst_telemetry

    def cross_validate_candidates(
        self,
        candidates: list[tuple[str, str]],  # (code, language)
        spec: ProblemSpec,
        inputs: list[str],
    ) -> list[tuple[str, int, ExecutionTelemetry]]:
        """
        Run multiple candidates across test inputs and calculate consensus voting.
        Returns list of (code, consensus_score, worst_telemetry).
        """
        results = []
        for code, lang in candidates:
            all_passed = True
            worst_telem = ExecutionTelemetry(
                verdict="AC",
                wall_clock_ms=0.0,
                cpu_time_ms=0.0,
                peak_rss_kb=0,
                exit_code=0,
                sandbox_engine=self.sandbox.engine,
            )
            outputs = []

            for inp in inputs:
                telem = self.sandbox.run_code(
                    code=code,
                    language="cpp" if lang == "cpp" else "python",
                    input_data=inp,
                    time_limit_ms=spec.time_limit_ms,
                    memory_limit_mb=spec.memory_limit_mb,
                )
                if telem.verdict != "AC":
                    all_passed = False
                    worst_telem.verdict = telem.verdict
                    break
                outputs.append(normalize_output(telem.stdout))
                worst_telem.cpu_time_ms = max(worst_telem.cpu_time_ms, telem.cpu_time_ms)
                worst_telem.peak_rss_kb = max(worst_telem.peak_rss_kb, telem.peak_rss_kb)

            if all_passed:
                results.append((code, outputs, worst_telem))

        # Score candidates based on output consensus across peers
        scored_candidates = []
        for i, (code_i, out_i, telem_i) in enumerate(results):
            score = 1
            for j, (code_j, out_j, _) in enumerate(results):
                if i != j and out_i == out_j:
                    score += 1
            scored_candidates.append((code_i, score, telem_i))

        # Sort candidates descending by consensus score, ascending by CPU time
        scored_candidates.sort(key=lambda item: (-item[1], item[2].cpu_time_ms))
        return scored_candidates


class FloatToleranceChecker:
    """Checker for problems with floating point outputs (e.g. within 1e-6 relative or absolute error)."""

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    def __call__(self, user_out: str, expected_out: str) -> bool:
        u_tokens = user_out.strip().split()
        e_tokens = expected_out.strip().split()

        if len(u_tokens) != len(e_tokens):
            return False

        for u, e in zip(u_tokens, e_tokens):
            try:
                u_f = float(u)
                e_f = float(e)
                diff = abs(u_f - e_f)
                denom = max(1.0, abs(e_f))
                if diff > self.tolerance and (diff / denom) > self.tolerance:
                    return False
            except ValueError:
                if u != e:
                    return False
        return True
