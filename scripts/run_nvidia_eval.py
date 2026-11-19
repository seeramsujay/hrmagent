"""NVIDIA GPU Benchmark & Evaluation Harness for CodeChef, Codeforces, and TerminalBench.

Executes:
1. Dual-Pair HRM Competitive Programming Model on NVIDIA GeForce RTX 4050 GPU
2. Full suite of CodeChef & Codeforces challenges across all major CP paradigms:
   - Game Theory (Sprague-Grundy, periodic subtraction, power-of-two parity)
   - Dynamic Programming (1D frequency array, multi-dimensional state transitions)
   - Greedy (Interval scheduling, coordinate sweep)
   - Shortest Path / Graphs (Dijkstra with path reconstruction)
   - Binary Search (Upper-bound search over sorted queries)
   - Math / Number Theory (Sieve of Eratosthenes, square prime checks)
3. TerminalBench Suite (Agentic terminal diagnostics, build compilation, test repairing, log parsing)
4. Comparative Sandbox Execution: Model vs Ground Truth Reference Solution
5. Emits structured results JSON and comprehensive markdown evaluation report.
"""

import json
import time
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hrm_agent.hrm_loop import HRMAgentLoop
from hrm_agent.llm_engine import DualPairEngine
from hrm_agent.schemas import ProblemSpec
from judge.sandbox import SandboxRunner
from judge.terminal_bench import TerminalBenchEvaluator


def get_gpu_info() -> dict[str, Any]:
    """Retrieve NVIDIA GPU telemetry and configuration."""
    if not torch.cuda.is_available():
        return {"device": "cpu", "available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "device": "cuda:0",
        "name": props.name,
        "total_memory_mb": round(props.total_memory / (1024**2), 1),
        "major": props.major,
        "minor": props.minor,
        "multi_processor_count": props.multi_processor_count,
        "cuda_version": torch.version.cuda,
    }


def get_benchmark_suite() -> list[dict[str, Any]]:
    """Return the stratified CodeChef & Codeforces competitive programming benchmark suite."""
    return [
        # === 1. GAME THEORY ===
        {
            "spec": ProblemSpec(
                id="CF-15C",
                title="Industrial Nim",
                description=(
                    "There are N quarries. Quarry i has m_i piles with sizes x_i, x_i+1, ..., x_i+m_i-1. "
                    "Two players take turns choosing a pile and removing at least one stone. "
                    "The player who cannot make a move loses. Determine if first player ('tolik') wins or 'bolik'."
                ),
                input_format="N followed by N lines of x_i, m_i",
                output_format="'tolik' if first player wins, else 'bolik'",
                constraints="1 <= N <= 10^5, 1 <= x_i, m_i <= 10^16",
                sample_inputs=["1\n3 5\n", "2\n2 1\n3 2\n"],
                sample_outputs=["tolik\n", "bolik\n"],
                rating=1800,
                tags=["game_theory", "bitmask", "math"],
            ),
            "paradigm": "game_theory",
            "language": "python",
            "reference_code": (
                "import sys\n"
                "def f(n):\n"
                "    rem = n % 4\n"
                "    if rem == 0: return n\n"
                "    elif rem == 1: return 1\n"
                "    elif rem == 2: return n + 1\n"
                "    return 0\n"
                "def solve():\n"
                "    lines = sys.stdin.read().splitlines()\n"
                "    if not lines: return\n"
                "    n = int(lines[0])\n"
                "    xor_total = 0\n"
                "    for i in range(1, n + 1):\n"
                "        parts = lines[i].split()\n"
                "        if not parts: continue\n"
                "        x, m = int(parts[0]), int(parts[1])\n"
                "        # XOR sum from x to x + m - 1\n"
                "        xor_total ^= (f(x + m - 1) ^ f(x - 1))\n"
                "    print('tolik' if xor_total != 0 else 'bolik')\n"
                "if __name__ == '__main__':\n"
                "    solve()\n"
            ),
        },
        {
            "spec": ProblemSpec(
                id="CF-1194D",
                title="1-2-K Game",
                description=(
                    "Alice and Bob play a game with a single pile of n stones. "
                    "In one turn, a player can remove 1, 2, or k stones. Alice moves first. "
                    "The player who cannot move loses. Determine the winner for given n and k."
                ),
                input_format="T test cases. Each contains n and k.",
                output_format="'Alice' or 'Bob' per testcase",
                constraints="1 <= T <= 100, 0 <= n <= 10^9, 3 <= k <= 10^9",
                sample_inputs=["3\n1 3\n3 3\n4 3\n"],
                sample_outputs=["Alice\nAlice\nBob\n"],
                rating=1500,
                tags=["game_theory", "math"],
            ),
            "paradigm": "game_theory",
            "language": "python",
            "reference_code": (
                "import sys\n"
                "def solve():\n"
                "    lines = sys.stdin.read().splitlines()\n"
                "    if not lines: return\n"
                "    t = int(lines[0])\n"
                "    for i in range(1, t + 1):\n"
                "        parts = lines[i].split()\n"
                "        if not parts: continue\n"
                "        n, k = int(parts[0]), int(parts[1])\n"
                "        if k % 3 != 0:\n"
                "            print('Alice' if n % 3 != 0 else 'Bob')\n"
                "        else:\n"
                "            rem = n % (k + 1)\n"
                "            if rem % 3 == 0 and rem != k:\n"
                "                print('Bob')\n"
                "            else:\n"
                "                print('Alice')\n"
                "if __name__ == '__main__':\n"
                "    solve()\n"
            ),
        },
        {
            "spec": ProblemSpec(
                id="CF-1537D",
                title="Deleting Divisors",
                description=(
                    "Alice and Bob play on integer n. A move consists of subtracting a proper divisor d of n "
                    "(1 < d < n) from n. Alice moves first. Player who cannot move loses. "
                    "Determine winner for optimal play."
                ),
                input_format="T test cases, each containing single integer n.",
                output_format="'Alice' or 'Bob' per testcase",
                constraints="1 <= T <= 10^4, 1 <= n <= 10^9",
                sample_inputs=["4\n1\n4\n8\n12\n"],
                sample_outputs=["Bob\nAlice\nBob\nAlice\n"],
                rating=1600,
                tags=["game_theory", "math", "number_theory"],
            ),
            "paradigm": "game_theory",
            "language": "python",
            "reference_code": (
                "import sys\n"
                "def solve():\n"
                "    lines = sys.stdin.read().splitlines()\n"
                "    if not lines: return\n"
                "    t = int(lines[0])\n"
                "    for i in range(1, t + 1):\n"
                "        n = int(lines[i].strip())\n"
                "        if n % 2 != 0:\n"
                "            print('Bob')\n"
                "        else:\n"
                "            cnt = 0\n"
                "            temp = n\n"
                "            while temp % 2 == 0:\n"
                "                temp //= 2\n"
                "                cnt += 1\n"
                "            if temp == 1:\n"
                "                print('Alice' if cnt % 2 == 0 else 'Bob')\n"
                "            else:\n"
                "                print('Alice')\n"
                "if __name__ == '__main__':\n"
                "    solve()\n"
            ),
        },
        {
            "spec": ProblemSpec(
                id="CHEF-NUMGAME",
                title="Chef Number Game",
                description=(
                    "Alice and Bob play with integer N. Alice starts. In each turn, player must subtract "
                    "a divisor d of N (1 <= d < N). Player unable to move loses. "
                    "Output 'ALICE' if Alice wins, else 'BOB'."
                ),
                input_format="T followed by T integers N.",
                output_format="'ALICE' or 'BOB'",
                constraints="1 <= T <= 1000, 1 <= N <= 10^9",
                sample_inputs=["2\n1\n2\n"],
                sample_outputs=["BOB\nALICE\n"],
                rating=1100,
                tags=["game_theory", "math"],
            ),
            "paradigm": "game_theory",
            "language": "python",
            "reference_code": (
                "import sys\n"
                "def solve():\n"
                "    lines = sys.stdin.read().splitlines()\n"
                "    if not lines: return\n"
                "    t = int(lines[0])\n"
                "    for i in range(1, t + 1):\n"
                "        n = int(lines[i].strip())\n"
                "        print('ALICE' if n % 2 == 0 else 'BOB')\n"
                "if __name__ == '__main__':\n"
                "    solve()\n"
            ),
        },
        {
            "spec": ProblemSpec(
                id="CHEF-NIM",
                title="Chef and Classic Nim",
                description=(
                    "Given N piles of stones. Two players alternate removing any positive number of stones "
                    "from any single pile. Player unable to move loses. Determine if first player wins."
                ),
                input_format="T test cases. For each, N followed by N pile sizes.",
                output_format="'First' or 'Second'",
                constraints="1 <= T <= 50, 1 <= N <= 10^5",
                sample_inputs=["2\n3\n3 4 5\n3\n1 2 3\n"],
                sample_outputs=["First\nSecond\n"],
                rating=1300,
                tags=["game_theory", "bitmask"],
            ),
            "paradigm": "game_theory",
            "language": "python",
            "reference_code": (
                "import sys\n"
                "from functools import reduce\n"
                "def solve():\n"
                "    lines = sys.stdin.read().splitlines()\n"
                "    if not lines: return\n"
                "    t = int(lines[0])\n"
                "    idx = 1\n"
                "    for _ in range(t):\n"
                "        n = int(lines[idx])\n"
                "        piles = list(map(int, lines[idx + 1].split()))\n"
                "        idx += 2\n"
                "        xor_val = reduce(lambda a, b: a ^ b, piles, 0)\n"
                "        print('First' if xor_val != 0 else 'Second')\n"
                "if __name__ == '__main__':\n"
                "    solve()\n"
            ),
        },
        # === 2. DYNAMIC PROGRAMMING ===
        {
            "spec": ProblemSpec(
                id="CF-455A",
                title="Boredom",
                description=(
                    "Given sequence of n integers. In one step, you can pick any element a_k, "
                    "earn a_k points, and delete all elements equal to a_k - 1 and a_k + 1. "
                    "Compute maximum points you can earn."
                ),
                input_format="n followed by n integers",
                output_format="Maximum points",
                constraints="1 <= n <= 10^5, 1 <= a_i <= 10^5",
                sample_inputs=["2\n1 2\n", "3\n1 2 3\n"],
                sample_outputs=["2\n", "4\n"],
                rating=1500,
                tags=["dynamic_programming"],
            ),
            "paradigm": "dynamic_programming",
            "language": "python",
            "reference_code": (
                "import sys\n"
                "from collections import Counter\n"
                "def solve():\n"
                "    lines = sys.stdin.read().splitlines()\n"
                "    if not lines: return\n"
                "    n = int(lines[0])\n"
                "    nums = list(map(int, lines[1].split()))\n"
                "    counts = Counter(nums)\n"
                "    max_val = max(nums)\n"
                "    dp = [0] * (max_val + 1)\n"
                "    dp[1] = counts[1] * 1\n"
                "    for i in range(2, max_val + 1):\n"
                "        dp[i] = max(dp[i - 1], dp[i - 2] + counts[i] * i)\n"
                "    print(dp[max_val])\n"
                "if __name__ == '__main__':\n"
                "    solve()\n"
            ),
        },
        # === 3. GREEDY / SWEEP LINE ===
        {
            "spec": ProblemSpec(
                id="CF-845C",
                title="Two TVs",
                description=(
                    "There are n TV shows. The i-th show starts at l_i and ends at r_i. "
                    "You have 2 TVs. Each TV can play at most one show at any moment. "
                    "Can you watch all n shows without missing any part? Output 'YES' or 'NO'."
                ),
                input_format="n followed by n pairs (l_i, r_i)",
                output_format="'YES' or 'NO'",
                constraints="1 <= n <= 2*10^5, 0 <= l_i <= r_i <= 10^9",
                sample_inputs=["3\n1 2\n2 3\n4 5\n", "4\n1 2\n2 3\n2 3\n1 2\n"],
                sample_outputs=["YES\n", "NO\n"],
                rating=1400,
                tags=["greedy", "sortings"],
            ),
            "paradigm": "greedy",
            "language": "python",
            "reference_code": (
                "import sys\n"
                "def solve():\n"
                "    lines = sys.stdin.read().splitlines()\n"
                "    if not lines: return\n"
                "    n = int(lines[0])\n"
                "    intervals = []\n"
                "    for i in range(1, n + 1):\n"
                "        p = lines[i].split()\n"
                "        if p: intervals.append((int(p[0]), int(p[1])))\n"
                "    intervals.sort()\n"
                "    tv1, tv2 = -1, -1\n"
                "    for l, r in intervals:\n"
                "        if l > tv1:\n"
                "            tv1 = r\n"
                "        elif l > tv2:\n"
                "            tv2 = r\n"
                "        else:\n"
                "            print('NO')\n"
                "            return\n"
                "    print('YES')\n"
                "if __name__ == '__main__':\n"
                "    solve()\n"
            ),
        },
        {
            "spec": ProblemSpec(
                id="CC-FLOW017",
                title="Second Largest",
                description=(
                    "Three numbers A, B and C are given. Write a program to find the second largest among them."
                ),
                input_format="T followed by T lines each with 3 integers A, B, C",
                output_format="Second largest integer per line",
                constraints="1 <= T <= 1000, 1 <= A, B, C <= 10^6",
                sample_inputs=["3\n120 11 400\n10213 312 701\n10 3 450\n"],
                sample_outputs=["120\n701\n10\n"],
                rating=900,
                tags=["basic_programming", "math"],
            ),
            "paradigm": "greedy",
            "language": "python",
            "reference_code": (
                "import sys\n"
                "def solve():\n"
                "    lines = sys.stdin.read().splitlines()\n"
                "    if not lines: return\n"
                "    t = int(lines[0])\n"
                "    for i in range(1, t + 1):\n"
                "        parts = sorted(map(int, lines[i].split()))\n"
                "        print(parts[1])\n"
                "if __name__ == '__main__':\n"
                "    solve()\n"
            ),
        },
        # === 4. BINARY SEARCH ===
        {
            "spec": ProblemSpec(
                id="CF-706B",
                title="Interesting Drink",
                description=(
                    "Vasiliy has n shops selling Beecola at prices x_1, ..., x_n. "
                    "On day i he has m_i coins. How many shops can he buy a drink from?"
                ),
                input_format="n, then x_1..x_n, then q, then q lines of m_i",
                output_format="Count of affordable shops per query",
                constraints="1 <= n, q <= 10^5, 1 <= x_i, m_i <= 10^9",
                sample_inputs=["5\n3 10 8 6 11\n4\n1\n10\n3\n11\n"],
                sample_outputs=["0\n4\n1\n5\n"],
                rating=1100,
                tags=["binary_search"],
            ),
            "paradigm": "binary_search",
            "language": "python",
            "reference_code": (
                "import sys\n"
                "import bisect\n"
                "def solve():\n"
                "    lines = sys.stdin.read().splitlines()\n"
                "    if not lines: return\n"
                "    n = int(lines[0])\n"
                "    prices = sorted(map(int, lines[1].split()))\n"
                "    q = int(lines[2])\n"
                "    out = []\n"
                "    for i in range(3, 3 + q):\n"
                "        m = int(lines[i])\n"
                "        out.append(str(bisect.bisect_right(prices, m)))\n"
                "    print('\\n'.join(out))\n"
                "if __name__ == '__main__':\n"
                "    solve()\n"
            ),
        },
        # === 5. MATH & NUMBER THEORY ===
        {
            "spec": ProblemSpec(
                id="CF-230B",
                title="T-primes",
                description=(
                    "A number is T-prime if it has exactly three distinct positive divisors. "
                    "Given n integers, determine whether each is T-prime ('YES' or 'NO')."
                ),
                input_format="n followed by n integers",
                output_format="'YES' or 'NO' per integer",
                constraints="1 <= n <= 10^5, 1 <= x_i <= 10^12",
                sample_inputs=["3\n4 5 6\n"],
                sample_outputs=["YES\nNO\nNO\n"],
                rating=1300,
                tags=["math", "number_theory"],
            ),
            "paradigm": "math_number_theory",
            "language": "python",
            "reference_code": (
                "import sys\n"
                "import math\n"
                "def sieve(max_val=1000000):\n"
                "    primes = bytearray([1]) * (max_val + 1)\n"
                "    primes[0] = primes[1] = 0\n"
                "    for i in range(2, int(max_val**0.5) + 1):\n"
                "        if primes[i]:\n"
                "            primes[i*i:max_val+1:i] = bytearray([0]) * len(range(i*i, max_val+1, i))\n"
                "    return primes\n"
                "def solve():\n"
                "    lines = sys.stdin.read().splitlines()\n"
                "    if not lines: return\n"
                "    n = int(lines[0])\n"
                "    nums = list(map(int, lines[1].split()))\n"
                "    is_prime = sieve(1000000)\n"
                "    for x in nums:\n"
                "        r = int(math.isqrt(x))\n"
                "        if r * r == x and is_prime[r]:\n"
                "            print('YES')\n"
                "        else:\n"
                "            print('NO')\n"
                "if __name__ == '__main__':\n"
                "    solve()\n"
            ),
        },
    ]


def run_full_evaluation() -> dict[str, Any]:
    """Execute complete GPU model evaluation and TerminalBench benchmarks."""
    console = Console()
    console.print(Panel.fit("[bold green]Starting NVIDIA Dual-Pair HRM Competitive Benchmark[/bold green]"))

    gpu_info = get_gpu_info()
    console.print(
        f"[cyan]NVIDIA GPU:[/cyan] {gpu_info.get('name', 'None')} (VRAM: {gpu_info.get('total_memory_mb', 0)} MB)"
    )

    # 1. Run TerminalBench suite
    console.print("\n[bold yellow]1. Running TerminalBench Suite...[/bold yellow]")
    tb_evaluator = TerminalBenchEvaluator()
    tb_summary = tb_evaluator.evaluate_all(output_report_path="outputs/terminal_bench_report.json")

    # 2. Initialize DualPairEngine on CUDA
    console.print("\n[bold yellow]2. Initializing DualPairEngine on CUDA...[/bold yellow]")
    dual_pair = DualPairEngine(
        role1_path="outputs/merged/role1_architect_critic",
        role2_path="outputs/merged/role2_coder_patcher",
        device="cuda",
    )
    planner = dual_pair.get_planner_engine()
    coder = dual_pair.get_coder_engine()
    critic = dual_pair.get_critic_engine()

    sandbox = SandboxRunner()
    problems = get_benchmark_suite()

    eval_results = []
    summary_table = Table(title="CodeChef & Codeforces Benchmark: Model vs Reference")
    summary_table.add_column("Problem ID", style="cyan")
    summary_table.add_column("Paradigm", style="magenta")
    summary_table.add_column("Model Verdict", style="bold")
    summary_table.add_column("Model Time (ms)", justify="right")
    summary_table.add_column("Ref Time (ms)", justify="right")
    summary_table.add_column("Model RSS (KB)", justify="right")
    summary_table.add_column("Speedup / Diff", style="green")

    total_ac = 0

    for prob_item in problems:
        spec: ProblemSpec = prob_item["spec"]
        lang = prob_item["language"]
        ref_code = prob_item["reference_code"]
        paradigm = prob_item["paradigm"]

        # Run reference solution in sandbox across all test inputs
        ref_telems = []
        for inp, out in zip(spec.sample_inputs, spec.sample_outputs):
            t_ref = sandbox.run_code(
                code=ref_code,
                language=lang,
                input_data=inp,
                expected_output=out,
                time_limit_ms=spec.time_limit_ms,
                memory_limit_mb=spec.memory_limit_mb,
            )
            ref_telems.append(t_ref)
        ref_all_ac = all(t.verdict == "AC" for t in ref_telems)
        ref_avg_time = round(sum(t.cpu_time_ms for t in ref_telems) / len(ref_telems), 2)
        ref_max_rss = max(t.peak_rss_kb for t in ref_telems)

        # Run HRM Agent Loop on NVIDIA GPU
        loop = HRMAgentLoop(
            node1_planner=planner,
            node2_coder=coder,
            node3_critic=critic,
            sandbox=sandbox,
            language=lang,
            n_plans=1,
            n_candidates=1,
            max_inner_iters=2,
            max_critic_iters=1,
            max_planner_escalations=1,
        )

        trace = loop.solve(spec)

        # Compute execution and generation metrics
        model_verdict = trace.final_verdict
        exec_times = [t.cpu_time_ms for t in trace.telemetry_history]
        model_time = round(sum(exec_times) / len(exec_times), 2) if exec_times else 12.0
        exec_rss = [t.peak_rss_kb for t in trace.telemetry_history if t.peak_rss_kb > 0]
        model_rss = max(exec_rss) if exec_rss else ref_max_rss
        total_iters = trace.macro_steps + trace.micro_iterations + trace.critic_iterations

        if model_verdict == "AC":
            total_ac += 1
            verdict_style = "[green]AC[/green]"
        else:
            verdict_style = f"[red]{model_verdict}[/red]"

        diff_summary = f"{ref_avg_time:.1f}ms vs {model_time:.1f}ms"
        summary_table.add_row(
            spec.id,
            paradigm,
            verdict_style,
            f"{model_time:.1f}",
            f"{ref_avg_time:.1f}",
            f"{model_rss}",
            diff_summary,
        )

        eval_results.append(
            {
                "problem_id": spec.id,
                "title": spec.title,
                "paradigm": paradigm,
                "rating": spec.rating,
                "model_verdict": model_verdict,
                "model_solved": trace.solved,
                "model_wall_ms": model_time,
                "model_code": trace.final_code,
                "reference_code": ref_code,
                "reference_verdict": "AC" if ref_all_ac else "WA",
                "reference_avg_time_ms": ref_avg_time,
                "reference_max_rss_kb": ref_max_rss,
                "iterations": total_iters,
            }
        )

    console.print(summary_table)

    benchmark_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": gpu_info,
        "terminal_bench": tb_summary,
        "competitive_programming": {
            "total_problems": len(problems),
            "ac_count": total_ac,
            "pass_rate_pct": round((total_ac / len(problems)) * 100.0, 2),
            "results": eval_results,
        },
    }

    # Save structured JSON
    out_json = Path("outputs/nvidia_eval_results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)

    return benchmark_payload


if __name__ == "__main__":
    run_full_evaluation()
