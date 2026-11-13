"""Stratified Benchmark Evaluator for CodeChef and Codeforces.

Evaluates Pass@1 across rating buckets (<1200, 1200-1500, 1500-1800, 1800-2100, 2100+),
runs 3-way ablation (Node 2 Only, Node 1+2, Full HRM), computes Jev precision & false-positive rate,
average CPU wall time, and token economics.
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from rich.console import Console
from rich.table import Table

from hrm_agent.schemas import ProblemSpec, AgentTrace, JevVerdict
from hrm_agent.hrm_loop import HRMAgentLoop
from hrm_agent.llm_engine import LLMEngine
from judge.sandbox import SandboxRunner


RATING_BUCKETS = [
    ("<1200", lambda r: r is not None and r < 1200),
    ("1200-1500", lambda r: r is not None and 1200 <= r < 1500),
    ("1500-1800", lambda r: r is not None and 1500 <= r < 1800),
    ("1800-2100", lambda r: r is not None and 1800 <= r < 2100),
    ("2100+", lambda r: r is not None and r >= 2100),
]


class BenchmarkEvaluator:
    """Evaluates competitive programming agents across stratified datasets."""

    def __init__(
        self,
        node1_planner: LLMEngine,
        node2_coder: LLMEngine,
        node3_critic: LLMEngine,
        sandbox: Optional[SandboxRunner] = None,
        language: str = "cpp",
        grammar_dir: str = "grammars",
    ):
        self.planner = node1_planner
        self.coder = node2_coder
        self.critic = node3_critic
        self.sandbox = sandbox or SandboxRunner()
        self.language = language
        self.grammar_dir = grammar_dir
        self.console = Console()

    def _get_bucket_label(self, rating: Optional[int]) -> str:
        if rating is None:
            return "1200-1500"
        for label, predicate in RATING_BUCKETS:
            if predicate(rating):
                return label
        return "1200-1500"

    def evaluate_ablation(
        self, problems: List[ProblemSpec], output_report_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run 3-way ablation comparison:
        1. Node 2 Only (Zero-shot fast coder)
        2. Node 1 + 2 (Planner + Coder)
        3. Full HRM (Node 1 Planner + Node 2 Coder + Node 3 Jev Critic with Escalation)
        """
        results: Dict[str, Any] = {
            "node2_only": {"solved": 0, "total": len(problems), "traces": []},
            "node1_2": {"solved": 0, "total": len(problems), "traces": []},
            "full_hrm": {"solved": 0, "total": len(problems), "traces": []},
            "bucket_stats": {label: {"total": 0, "hrm_solved": 0} for label, _ in RATING_BUCKETS},
            "jev_metrics": {
                "true_positives": 0,
                "false_positives": 0,
                "true_negatives": 0,
                "false_negatives": 0,
            },
            "total_tokens": 0,
            "total_wall_ms": 0.0,
        }

        for prob in problems:
            bucket = self._get_bucket_label(prob.rating)
            results["bucket_stats"][bucket]["total"] += 1

            # Config 1: Node 2 Only (No planner, no critic)
            loop_n2 = HRMAgentLoop(
                node1_planner=self.planner,
                node2_coder=self.coder,
                node3_critic=self.critic,
                sandbox=self.sandbox,
                language=self.language,
                n_plans=0,
                n_candidates=1,
                max_inner_iters=0,
                max_critic_iters=0,
                max_planner_escalations=0,
                grammar_dir=self.grammar_dir,
            )
            # Direct code generation with dummy blueprint
            dummy_bp = None
            code_n2, telem_n2 = loop_n2._run_coder_plan_to_code(
                prob,
                blueprint=type(
                    "DummyBP",
                    (),
                    {"model_dump_json": lambda self: json.dumps({"paradigm": "direct_implementation"})},
                )(),
            )
            passed_n2, _, telem_run_n2 = loop_n2._test_against_samples(code_n2, prob)
            if passed_n2:
                results["node2_only"]["solved"] += 1
            results["node2_only"]["traces"].append({"problem_id": prob.id, "solved": passed_n2})

            # Config 2: Node 1 + 2 (Planner + Coder, max_inner_iters=3, no critic)
            loop_n1_2 = HRMAgentLoop(
                node1_planner=self.planner,
                node2_coder=self.coder,
                node3_critic=self.critic,
                sandbox=self.sandbox,
                language=self.language,
                n_plans=1,
                n_candidates=2,
                max_inner_iters=3,
                max_critic_iters=0,
                max_planner_escalations=0,
                grammar_dir=self.grammar_dir,
            )
            trace_n1_2 = loop_n1_2.solve(prob)
            if trace_n1_2.solved:
                results["node1_2"]["solved"] += 1
            results["node1_2"]["traces"].append({"problem_id": prob.id, "solved": trace_n1_2.solved})

            # Config 3: Full HRM (Node 1 + Node 2 + Node 3 Jev Critic with Escalation)
            loop_full = HRMAgentLoop(
                node1_planner=self.planner,
                node2_coder=self.coder,
                node3_critic=self.critic,
                sandbox=self.sandbox,
                language=self.language,
                n_plans=1,
                n_candidates=3,
                max_inner_iters=3,
                max_critic_iters=2,
                max_planner_escalations=1,
                grammar_dir=self.grammar_dir,
            )
            trace_full = loop_full.solve(prob)
            if trace_full.solved:
                results["full_hrm"]["solved"] += 1
                results["bucket_stats"][bucket]["hrm_solved"] += 1
            results["full_hrm"]["traces"].append(trace_full.model_dump())

            # Track tokens & wall time
            prob_tokens = sum(m.completion_tokens + m.prompt_tokens for m in trace_full.llm_metrics)
            prob_wall = sum(t.wall_clock_ms for t in trace_full.telemetry_history)
            results["total_tokens"] += prob_tokens
            results["total_wall_ms"] += prob_wall

            # Evaluate Jev Critic Accuracy
            for verdict in trace_full.critic_verdicts:
                is_actually_valid = trace_full.solved and trace_full.final_verdict == "AC"
                if verdict.status in ("BUG", "OPTIMIZE"):
                    if not is_actually_valid:
                        results["jev_metrics"]["true_positives"] += 1
                    else:
                        results["jev_metrics"]["false_positives"] += 1
                else:  # verdict.status == "PASS"
                    if is_actually_valid:
                        results["jev_metrics"]["true_negatives"] += 1
                    else:
                        results["jev_metrics"]["false_negatives"] += 1

        # Summary calculations
        n_probs = len(problems) if problems else 1
        results["pass_at_1"] = {
            "node2_only": round((results["node2_only"]["solved"] / n_probs) * 100, 2),
            "node1_2": round((results["node1_2"]["solved"] / n_probs) * 100, 2),
            "full_hrm": round((results["full_hrm"]["solved"] / n_probs) * 100, 2),
        }
        results["avg_cpu_wall_ms"] = round(results["total_wall_ms"] / n_probs, 2)
        results["avg_tokens_per_problem"] = round(results["total_tokens"] / n_probs, 1)

        # Jev metrics
        tp = results["jev_metrics"]["true_positives"]
        fp = results["jev_metrics"]["false_positives"]
        tn = results["jev_metrics"]["true_negatives"]
        results["jev_precision"] = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 1.0
        results["jev_false_positive_rate"] = round(fp / (fp + tn), 3) if (fp + tn) > 0 else 0.0

        if output_report_path:
            Path(output_report_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_report_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

        return results

    def print_summary_tables(self, results: Dict[str, Any]) -> None:
        """Display rich terminal tables for Pass@1 ablation and rating buckets."""
        # Table 1: Ablation Comparison
        abl_table = Table(title="HRM Competitive Programming Agent — Ablation Suite")
        abl_table.add_column("Configuration", style="cyan")
        abl_table.add_column("Solved / Total", justify="center")
        abl_table.add_column("Pass@1 (%)", style="bold green", justify="right")

        abl_table.add_row(
            "Node 2 Only (Zero-Shot Coder)",
            f"{results['node2_only']['solved']} / {results['node2_only']['total']}",
            f"{results['pass_at_1']['node2_only']}%",
        )
        abl_table.add_row(
            "Node 1 + Node 2 (Planner + Coder)",
            f"{results['node1_2']['solved']} / {results['node1_2']['total']}",
            f"{results['pass_at_1']['node1_2']}%",
        )
        abl_table.add_row(
            "Full HRM (Node 1 + 2 + Jev Critic)",
            f"{results['full_hrm']['solved']} / {results['full_hrm']['total']}",
            f"{results['pass_at_1']['full_hrm']}%",
        )
        self.console.print(abl_table)

        # Table 2: Rating Buckets Stratification
        bucket_table = Table(title="Pass@1 Stratified by Difficulty Rating")
        bucket_table.add_column("Rating Bucket", style="magenta")
        bucket_table.add_column("Problems", justify="center")
        bucket_table.add_column("Solved", justify="center")
        bucket_table.add_column("Pass@1 (%)", style="green", justify="right")

        for bucket, stats in results["bucket_stats"].items():
            tot = stats["total"]
            sol = stats["hrm_solved"]
            pct = round((sol / tot) * 100, 1) if tot > 0 else 0.0
            bucket_table.add_row(bucket, str(tot), str(sol), f"{pct}%")
        self.console.print(bucket_table)

        # Table 3: Jev Critic & Telemetry Metrics
        tele_table = Table(title="Jev Critic & Efficiency Telemetry")
        tele_table.add_column("Metric", style="yellow")
        tele_table.add_column("Value", style="bold", justify="right")

        tele_table.add_row("Jev Critic Precision", f"{results['jev_precision']:.3f}")
        tele_table.add_row("Jev Critic False Positive Rate", f"{results['jev_false_positive_rate']:.3f}")
        tele_table.add_row("Avg Execution Wall Time", f"{results['avg_cpu_wall_ms']} ms")
        tele_table.add_row("Avg Tokens / Problem", f"{results['avg_tokens_per_problem']}")
        self.console.print(tele_table)
