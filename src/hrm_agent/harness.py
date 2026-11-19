"""Evaluation Harness Client and Benchmark Runner for HRM Agent.

Enables automated benchmarking harnesses to interface with the localhost HRM server,
verifying:
1. Standard OpenAI chat completion protocol (/v1/chat/completions)
2. Closed-loop multi-node HRM competitive programming solve protocol (/solve)
3. End-to-end Pass@1, verdict distributions, and execution latency.
"""

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hrm_agent.schemas import ProblemSpec

console = Console()


class HRMHarnessClient:
    """HTTP client targeting local HRM Agent server."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def _post(self, path: str, payload: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "HRM-Harness/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code} from {url}: {err_body}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to communicate with HRM server at {url}: {e}") from e

    def _get(self, path: str, timeout: float = 30.0) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, headers={"User-Agent": "HRM-Harness/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"Failed GET request to {url}: {e}") from e

    def check_health(self) -> dict[str, Any]:
        """Verify server health and retrieve GPU/model metadata."""
        return self._get("/health")

    def list_models(self) -> list[dict[str, Any]]:
        """Retrieve OpenAI-compatible model cards."""
        res = self._get("/v1/models")
        return res.get("data", [])

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str = "hrm-dual-pair",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Send chat completion query via standard OpenAI API format."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        return self._post("/v1/chat/completions", payload)

    def solve_problem(
        self,
        spec: ProblemSpec | dict[str, Any],
        language: str = "python",
        timeout: float = 180.0,
    ) -> dict[str, Any]:
        """Send competitive programming problem specification to /solve."""
        payload = spec.model_dump() if isinstance(spec, ProblemSpec) else spec
        return self._post(f"/solve?language={language}", payload, timeout=timeout)


def load_harness_problems(
    problems_path: str = "data/codechef_raw.json",
    limit: int = 5,
    allow_synthetic: bool = True,
) -> list[ProblemSpec]:
    """Load benchmark problems from JSON or fall back to synthetic specimens."""
    specs: list[ProblemSpec] = []
    p = Path(problems_path)

    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data[:limit]:
                    raw_in = item.get("sample_inputs")
                    if raw_in is None:
                        raw_in = item.get("sample_input", [])
                    if isinstance(raw_in, str):
                        raw_in = [raw_in]
                    elif not isinstance(raw_in, list):
                        raw_in = []

                    raw_out = item.get("sample_outputs")
                    if raw_out is None:
                        raw_out = item.get("sample_output", [])
                    if isinstance(raw_out, str):
                        raw_out = [raw_out]
                    elif not isinstance(raw_out, list):
                        raw_out = []

                    time_limit = int(item.get("time_limit_ms") or item.get("time_limit") or 2000)
                    memory_limit = int(item.get("memory_limit_mb") or item.get("memory_limit") or 256)

                    spec = ProblemSpec(
                        id=str(item.get("id", item.get("problem_id", f"PROB-{len(specs) + 1}"))),
                        title=str(item.get("title", "")),
                        description=str(item.get("description", item.get("statement", ""))),
                        input_format=str(item.get("input_format", "")),
                        output_format=str(item.get("output_format", "")),
                        constraints=str(item.get("constraints", "")),
                        sample_inputs=raw_in,
                        sample_outputs=raw_out,
                        rating=item.get("rating"),
                        time_limit_ms=time_limit,
                        memory_limit_mb=memory_limit,
                    )
                    specs.append(spec)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not parse {problems_path}: {e}[/yellow]")

    if not specs:
        if not allow_synthetic:
            raise FileNotFoundError(
                f"Requested problem corpus at '{problems_path}' not found or contains no valid problems."
            )
        console.print(
            f"[yellow]Corpus at '{problems_path}' was unavailable or empty. "
            "Loading explicitly marked synthetic specimens for demonstration mode.[/yellow]"
        )
        # Fallback synthetic specimens explicitly tagged
        specs = [
            ProblemSpec(
                id="HARNESS-SYNTH-1",
                title="Number Mirror (Synthetic Benchmark Specimen)",
                description="Read a single integer N and output it.",
                input_format="Single integer N",
                output_format="Single integer N",
                constraints="0 <= N <= 10^5",
                sample_inputs=["123\n"],
                sample_outputs=["123\n"],
                time_limit_ms=2000,
                memory_limit_mb=256,
            ),
            ProblemSpec(
                id="HARNESS-SYNTH-2",
                title="Sum of Digits (Synthetic Benchmark Specimen)",
                description="Given integer T, followed by T integers, output sum of digits for each.",
                input_format="T, then integers.",
                output_format="Sum of digits.",
                constraints="1 <= T <= 100",
                sample_inputs=["2\n12345\n312\n"],
                sample_outputs=["15\n6\n"],
                time_limit_ms=2000,
                memory_limit_mb=256,
            ),
        ]
    return specs[:limit]


def run_benchmark_harness(
    server_url: str = "http://127.0.0.1:8000",
    problems_file: str = "data/codechef_raw.json",
    limit: int = 5,
    language: str = "python",
    report_path: str | None = "outputs/harness_report.json",
) -> dict[str, Any]:
    """Execute evaluation harness against HRM server."""
    client = HRMHarnessClient(base_url=server_url)

    console.print(Panel.fit(f"[bold cyan]Running HRM Evaluation Harness against {server_url}[/bold cyan]"))

    # 1. Health & Model Checks
    try:
        health = client.check_health()
        console.print(
            f"[green]Server Health OK | Mode: {health.get('mode')} | "
            f"CUDA: {health.get('cuda_available')} | GPU: {health.get('gpu_name')}[/green]"
        )
    except Exception as e:
        console.print(f"[bold red]Failed to connect to server: {e}[/bold red]")
        raise

    models = client.list_models()
    model_ids = [str(m.get("id", "")) for m in models]
    console.print(f"[cyan]Available Harness Models: {', '.join(model_ids)}[/cyan]")

    # 2. Test OpenAI Chat Completion Protocol
    console.print("\n[bold yellow]Testing OpenAI Chat Protocol (/v1/chat/completions)...[/bold yellow]")
    chat_resp = client.chat_completion(
        messages=[
            {"role": "system", "content": "You are an expert coder."},
            {"role": "user", "content": "Write a python function to add two numbers."},
        ],
        model="hrm-dual-pair",
        max_tokens=100,
    )
    answer = chat_resp["choices"][0]["message"]["content"]
    usage = chat_resp.get("usage", {})
    console.print(
        f"[green]OpenAI Chat Protocol Verified! Used {usage.get('total_tokens', 0)} tokens.[/green]\n"
        f"[dim]Sample response: {answer.strip()[:60]}...[/dim]"
    )

    # 3. Benchmark Problem Solving
    problems = load_harness_problems(problems_path=problems_file, limit=limit)
    console.print(f"\n[bold yellow]Evaluating {len(problems)} Challenges via /solve ({language})...[/bold yellow]")

    results = []
    solved_count = 0
    start_total = time.time()

    table = Table(title=f"HRM Harness Results ({language.upper()})")
    table.add_column("Problem ID", style="cyan")
    table.add_column("Verdict", style="bold")
    table.add_column("Solved", style="bold")
    table.add_column("Macro Steps", style="magenta")
    table.add_column("Micro Iters", style="magenta")
    table.add_column("Critic Iters", style="magenta")

    for spec in problems:
        t0 = time.time()
        try:
            trace = client.solve_problem(spec, language=language)
            duration = time.time() - t0
            verdict = trace.get("final_verdict", "UNKNOWN")
            solved = trace.get("solved", False)
            if solved:
                solved_count += 1

            results.append(
                {
                    "problem_id": spec.id,
                    "title": spec.title,
                    "solved": solved,
                    "verdict": verdict,
                    "macro_steps": trace.get("macro_steps", 0),
                    "micro_iterations": trace.get("micro_iterations", 0),
                    "critic_iterations": trace.get("critic_iterations", 0),
                    "duration_sec": round(duration, 2),
                    "code_snippet": (trace.get("final_code") or "")[:150],
                }
            )

            v_style = "green" if verdict == "AC" else "red"
            table.add_row(
                spec.id,
                f"[{v_style}]{verdict}[/{v_style}]",
                "[green]YES[/green]" if solved else "[red]NO[/red]",
                str(trace.get("macro_steps", 0)),
                str(trace.get("micro_iterations", 0)),
                str(trace.get("critic_iterations", 0)),
            )
        except Exception as e:
            console.print(f"[red]Error evaluating problem {spec.id}: {e}[/red]")
            results.append(
                {
                    "problem_id": spec.id,
                    "solved": False,
                    "verdict": "ERROR",
                    "error": str(e),
                }
            )
            table.add_row(spec.id, "[red]ERROR[/red]", "[red]NO[/red]", "-", "-", "-")

    total_time = round(time.time() - start_total, 2)
    pass_at_1 = round(solved_count / len(problems), 4) if problems else 0.0

    console.print(table)
    console.print(
        f"[bold green]Harness Completed: {solved_count}/{len(problems)} Solved "
        f"(Pass@1: {pass_at_1 * 100:.1f}%) in {total_time}s[/bold green]"
    )

    report = {
        "timestamp": time.time(),
        "server_url": server_url,
        "corpus_file": problems_file,
        "corpus_type": "file" if Path(problems_file).exists() else "synthetic_demonstration",
        "language": language,
        "total_problems": len(problems),
        "solved_count": solved_count,
        "pass_at_1": pass_at_1,
        "total_time_seconds": total_time,
        "results": results,
    }

    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        console.print(f"[dim]Harness report saved to {report_path}[/dim]")

    return report
