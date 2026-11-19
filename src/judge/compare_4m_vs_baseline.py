"""Side-by-side comparative evaluation: Baseline Sub-1B Dual Pair vs HRM-4M Architecture.

Evaluates 10 competitive programming challenges across 6 paradigms:
1. CHEF-NUMGAME (Parity Game Theory)
2. CF-15C (Industrial Nim - Sprague-Grundy & O(1) Prefix XOR)
3. CF-1194D (1-2-K Game - Modulo k+1 Cycle)
4. CF-1537D (Deleting Divisors - Prime Power Parity)
5. CHEF-NIM (Classic Multi-Pile Nim - XOR Theorem)
6. CF-455A (Boredom - 1D Dynamic Programming)
7. CF-845C (Two TVs - Greedy Interval Scheduling)
8. CC-FLOW017 (Second Largest - Median Selection)
9. CF-706B (Interesting Drink - Binary Search / Bisect)
10. CF-230B (T-primes - Sieve & Square-of-Prime Number Theory)
"""

import json
import os
import time
from typing import Any

from rich.console import Console
from rich.table import Table

from hrm_agent.four_model_agent import FourModelAgent
from hrm_agent.schemas import ProblemSpec

BENCHMARK_PROBLEMS = [
    ProblemSpec(
        id="CHEF-NUMGAME",
        title="Chef Number Game",
        description="Alice and Bob take turns subtracting a proper divisor from integer N. Parity game.",
        constraints="1 <= T <= 10, 1 <= N <= 10^9",
        sample_inputs=["2\n1\n2\n"],
        sample_outputs=["BOB\nALICE\n"],
        rating=1200,
    ),
    ProblemSpec(
        id="CF-15C",
        title="Industrial Nim",
        description="Quarry ranges with consecutive stone piles. Nim sum computed using O(1) prefix XOR.",
        constraints="1 <= N <= 10^5, 1 <= x_i, m_i <= 10^16",
        sample_inputs=["2\n2 1\n3 2\n"],
        sample_outputs=["tolik\n"],
        rating=1800,
    ),
    ProblemSpec(
        id="CF-1194D",
        title="1-2-K Game",
        description="Subtraction game with allowed moves 1, 2, or K stones. Modulo k+1 cycle game theory.",
        constraints="1 <= T <= 100, 1 <= n, k <= 10^9",
        sample_inputs=["3\n1 3\n3 3\n4 3\n"],
        sample_outputs=["Alice\nAlice\nBob\n"],
        rating=1400,
    ),
    ProblemSpec(
        id="CF-1537D",
        title="Deleting Divisors",
        description="Alice and Bob play on integer N. Subtract proper divisor. Game on prime power parity.",
        constraints="1 <= T <= 100, 1 <= N <= 10^9",
        sample_inputs=["4\n1\n4\n12\n9\n"],
        sample_outputs=["Bob\nAlice\nAlice\nBob\n"],
        rating=1500,
    ),
    ProblemSpec(
        id="CHEF-NIM",
        title="Classic Multi-Pile Nim",
        description="Players alternate taking stones from piles. Sprague-Grundy XOR sum theorem.",
        constraints="1 <= T <= 10, 1 <= N <= 10^5, 1 <= A[i] <= 10^9",
        sample_inputs=["2\n3\n3 4 5\n3\n1 2 3\n"],
        sample_outputs=["First\nSecond\n"],
        rating=1300,
    ),
    ProblemSpec(
        id="CF-455A",
        title="Boredom",
        description="Choose element a_k, delete all elements equal to a_k - 1 and a_k + 1. Maximize points via DP.",
        constraints="1 <= N <= 10^5, 1 <= a_i <= 10^5",
        sample_inputs=["2\n1 2\n"],
        sample_outputs=["2\n"],
        rating=1500,
    ),
    ProblemSpec(
        id="CF-845C",
        title="Two TVs",
        description="Check if Polycarp can watch all N TV shows using at most 2 TV sets. Greedy interval scheduling.",
        constraints="1 <= N <= 2*10^5, 0 <= l_i < r_i <= 10^9",
        sample_inputs=["3\n1 2\n2 3\n4 5\n"],
        sample_outputs=["YES\n"],
        rating=1500,
    ),
    ProblemSpec(
        id="CC-FLOW017",
        title="Second Largest",
        description="Three numbers A, B and C are given. Find the second largest among them.",
        constraints="1 <= T <= 100, 1 <= A, B, C <= 10^6",
        sample_inputs=["2\n10 20 30\n500 100 250\n"],
        sample_outputs=["20\n250\n"],
        rating=800,
    ),
    ProblemSpec(
        id="CF-706B",
        title="Interesting Drink",
        description="Vasiliy wants to buy a drink each day. Number of shops affordable given coin budget via bisect.",
        constraints="1 <= N <= 10^5, 1 <= x_i <= 10^5, 1 <= q <= 10^5",
        sample_inputs=["5\n3 10 8 6 11\n4\n1\n10\n3\n11\n"],
        sample_outputs=["0\n4\n1\n5\n"],
        rating=1100,
    ),
    ProblemSpec(
        id="CF-230B",
        title="T-primes",
        description="Number is T-prime if it has exactly three distinct positive divisors. Square of prime check.",
        constraints="1 <= N <= 10^5, 1 <= x_i <= 10^12",
        sample_inputs=["3\n4 5 6\n"],
        sample_outputs=["YES\nNO\nNO\n"],
        rating=1300,
    ),
]


def run_comparison(output_path: str = "outputs/hrm_4m_vs_baseline_comparison.json") -> dict[str, Any]:
    """Run side-by-side comparison of Baseline Sub-1B vs HRM-4M."""
    console = Console()
    console.print("[bold cyan]Running 4M vs Baseline Comparative Benchmark[/bold cyan]")

    # Load baseline results if present
    baseline_map: dict[str, dict[str, Any]] = {}
    if os.path.exists("outputs/nvidia_eval_results.json"):
        with open("outputs/nvidia_eval_results.json") as f:
            nv_data = json.load(f)
            for r in nv_data.get("competitive_programming", {}).get("results", []):
                baseline_map[r["problem_id"]] = r

    agent = FourModelAgent(device="auto")
    results: list[dict[str, Any]] = []

    table = Table(title="Baseline Sub-1B vs HRM-4M (Four-Model Hierarchical Reasoning)")
    table.add_column("Problem ID", style="bold")
    table.add_column("Title")
    table.add_column("Rating")
    table.add_column("Baseline 0.5B", justify="center")
    table.add_column("HRM-4M Verdict", justify="center")
    table.add_column("HRM-4M Time (ms)", justify="right")
    table.add_column("Latent Steps", justify="right")
    table.add_column("Invariant Discovered")

    baseline_ac_count = 0
    hrm_4m_ac_count = 0

    for prob in BENCHMARK_PROBLEMS:
        start_t = time.perf_counter()
        trace = agent.solve(prob, language="python")
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        # Baseline info
        base_res = baseline_map.get(prob.id, {})
        base_verdict = base_res.get("model_verdict", "RE")
        if base_verdict == "AC":
            baseline_ac_count += 1

        if trace.final_verdict == "AC":
            hrm_4m_ac_count += 1

        latent_steps = trace.latent_invariant.get("latent_steps", 0)
        inv_str = trace.latent_invariant.get("invariant", "")[:45]

        base_style = "[green]AC[/green]" if base_verdict == "AC" else f"[red]{base_verdict}[/red]"
        hrm_style = (
            "[bold green]AC[/bold green]"
            if trace.final_verdict == "AC"
            else f"[bold red]{trace.final_verdict}[/bold red]"
        )

        table.add_row(
            prob.id,
            prob.title[:20],
            str(prob.rating or "-"),
            base_style,
            hrm_style,
            f"{elapsed_ms:.1f}",
            str(latent_steps),
            inv_str,
        )

        results.append(
            {
                "problem_id": prob.id,
                "title": prob.title,
                "rating": prob.rating,
                "baseline_verdict": base_verdict,
                "hrm_4m_verdict": trace.final_verdict,
                "hrm_4m_solved": trace.solved,
                "hrm_4m_wall_ms": elapsed_ms,
                "latent_steps": latent_steps,
                "latent_invariant": trace.latent_invariant,
                "synthesized_code": trace.synthesized_code,
            }
        )

    console.print(table)
    summary = {
        "total_problems": len(BENCHMARK_PROBLEMS),
        "baseline_ac_count": baseline_ac_count,
        "baseline_pass_rate_pct": (baseline_ac_count / len(BENCHMARK_PROBLEMS)) * 100.0,
        "hrm_4m_ac_count": hrm_4m_ac_count,
        "hrm_4m_pass_rate_pct": (hrm_4m_ac_count / len(BENCHMARK_PROBLEMS)) * 100.0,
        "results": results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    console.print(
        f"\n[bold green]Benchmark Complete![/bold green] "
        f"Baseline: {baseline_ac_count}/{len(BENCHMARK_PROBLEMS)} ({summary['baseline_pass_rate_pct']}%) -> "
        f"HRM-4M: {hrm_4m_ac_count}/{len(BENCHMARK_PROBLEMS)} ({summary['hrm_4m_pass_rate_pct']}%)\n"
    )
    return summary


if __name__ == "__main__":
    run_comparison()
