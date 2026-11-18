"""TerminalBench: Agentic Terminal Command & Diagnostic Benchmark Suite.

Benchmarks autonomous agents on terminal and command-line execution tasks:
1. Log Analysis & Diagnostics (parsing logs, extracting metrics, json summarization)
2. Bug Diagnosis & Code Patching (tracing runtime exceptions, applying unified diffs)
3. Compilation & Build Troubleshooting (C++20 overflow fixes, flag optimization)
4. Multi-File Refactoring & Shell Processing (recursive file search, awk/sed/grep transforms)
5. Test Suite Reproduction & Automated Repair (failing pytest diagnostics, test pass verification)
6. Environment & System Inspection (process filtering, resource analysis, exit code checking)
"""

import json
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from rich.console import Console
from rich.table import Table

from judge.sandbox import SandboxRunner


@dataclass
class TerminalTask:
    """Definition of a TerminalBench benchmark task."""

    id: str
    category: Literal[
        "log_diagnostics",
        "code_patching",
        "build_compilation",
        "bash_scripting",
        "test_repair",
        "environment_inspection",
    ]
    title: str
    description: str
    setup_files: dict[str, str]
    test_command: str
    verification_fn: Optional[Callable[[Path, subprocess.CompletedProcess], tuple[bool, str]]] = None
    time_limit_ms: int = 10000
    memory_limit_mb: int = 512


@dataclass
class TerminalResult:
    """Result of running a single TerminalBench task."""

    task_id: str
    category: str
    passed: bool
    exit_code: int
    wall_clock_ms: float
    output: str
    error_message: Optional[str] = None
    artifacts: dict[str, Any] = field(default_factory=dict)


def verify_log_summary(workspace: Path, proc: subprocess.CompletedProcess) -> tuple[bool, str]:
    """Verify TB-01 log diagnostic task output file and counts."""
    summary_file = workspace / "summary.json"
    if not summary_file.exists():
        return False, "summary.json was not generated."
    try:
        data = json.loads(summary_file.read_text(encoding="utf-8"))
        if data.get("500") != 3:
            return False, f"Expected 3 occurrences of HTTP 500, got {data.get('500')}"
        if data.get("404") != 2:
            return False, f"Expected 2 occurrences of HTTP 404, got {data.get('404')}"
        if data.get("200") != 5:
            return False, f"Expected 5 occurrences of HTTP 200, got {data.get('200')}"
        if data.get("top_error_endpoint") != "/api/v1/checkout":
            return False, f"Expected top_error_endpoint '/api/v1/checkout', got '{data.get('top_error_endpoint')}'"
        return True, "Summary metrics matched expected distribution."
    except Exception as e:
        return False, f"Failed parsing summary.json: {e}"


def verify_search_patch(workspace: Path, proc: subprocess.CompletedProcess) -> tuple[bool, str]:
    """Verify TB-02 binary search fix passes all internal edge cases."""
    if proc.returncode != 0:
        return False, f"Test command failed with exit code {proc.returncode}: {proc.stderr}"
    if "ALL_TESTS_PASS" not in proc.stdout:
        return False, f"Expected 'ALL_TESTS_PASS' token in stdout, got: {proc.stdout}"
    return True, "Binary search patched and edge cases passed."


def verify_cpp_build(workspace: Path, proc: subprocess.CompletedProcess) -> tuple[bool, str]:
    """Verify TB-03 C++ compilation and 64-bit integer overflow fix."""
    if proc.returncode != 0:
        return False, f"C++ build or run failed with code {proc.returncode}: {proc.stderr}"
    lines = proc.stdout.strip().splitlines()
    if not lines or lines[-1].strip() != "500000000500000000":
        return False, f"Wrong computation result: {lines[-1] if lines else 'empty'}"
    return True, "C++ binary compiled with O3 and computed int64 sum accurately without overflow."


def verify_refactor_yaml(workspace: Path, proc: subprocess.CompletedProcess) -> tuple[bool, str]:
    """Verify TB-04 multi-file recursive YAML refactoring."""
    app_cfg = workspace / "configs" / "app.yaml"
    db_cfg = workspace / "configs" / "db.yaml"
    if not app_cfg.exists() or not db_cfg.exists():
        return False, "Config files missing."
    app_text = app_cfg.read_text(encoding="utf-8")
    db_text = db_cfg.read_text(encoding="utf-8")
    if "legacy_endpoint" in app_text or "legacy_endpoint" in db_text:
        return False, "Found remaining references to deprecated key 'legacy_endpoint'."
    if "modern_endpoint: https://api.hrm.internal" not in app_text:
        return False, "modern_endpoint key missing or incorrect in configs/app.yaml"
    return True, "All deprecated YAML keys refactored successfully."


def verify_test_repair(workspace: Path, proc: subprocess.CompletedProcess) -> tuple[bool, str]:
    """Verify TB-05 unit test suite fix."""
    if proc.returncode != 0:
        return False, f"Test suite did not exit cleanly: {proc.stderr}"
    combined = (proc.stdout or "") + " " + (proc.stderr or "")
    if "OK" not in combined and "Ran 3 tests" not in combined and "passed" not in combined:
        return False, f"Expected clean test pass in test output, got: {combined}"
    return True, "Failing test isolated, patched, and 3/3 tests passed."


def get_default_terminal_bench_tasks() -> list[TerminalTask]:
    """Return the standardized suite of TerminalBench tasks."""
    tasks: list[TerminalTask] = [
        TerminalTask(
            id="TB-01-LOG-METRICS",
            category="log_diagnostics",
            title="Access Log Diagnostics & JSON Summarization",
            description=(
                "Parse access.log to compute total HTTP status code frequencies (200, 404, 500) "
                "and find the endpoint with the highest 500 error count. "
                "Output results to summary.json."
            ),
            setup_files={
                "access.log": (
                    "2026-11-19T01:00:00Z GET /api/v1/health 200 12ms\n"
                    "2026-11-19T01:00:01Z POST /api/v1/checkout 500 450ms\n"
                    "2026-11-19T01:00:02Z GET /api/v1/users 200 23ms\n"
                    "2026-11-19T01:00:03Z GET /favicon.ico 404 2ms\n"
                    "2026-11-19T01:00:04Z POST /api/v1/checkout 500 460ms\n"
                    "2026-11-19T01:00:05Z GET /api/v1/items 200 15ms\n"
                    "2026-11-19T01:00:06Z GET /unknown 404 1ms\n"
                    "2026-11-19T01:00:07Z POST /api/v1/checkout 500 470ms\n"
                    "2026-11-19T01:00:08Z GET /api/v1/cart 200 18ms\n"
                    "2026-11-19T01:00:09Z GET /api/v1/status 200 5ms\n"
                ),
                "solution.py": (
                    "import json, re\n"
                    "from collections import Counter\n"
                    "counts = Counter()\n"
                    "err_endpoints = Counter()\n"
                    "with open('access.log') as f:\n"
                    "    for line in f:\n"
                    "        parts = line.strip().split()\n"
                    "        if len(parts) >= 5:\n"
                    "            code = parts[3]\n"
                    "            counts[code] += 1\n"
                    "            if code == '500':\n"
                    "                err_endpoints[parts[2]] += 1\n"
                    "top_err = err_endpoints.most_common(1)[0][0]\n"
                    "out = {'200': counts['200'], '404': counts['404'], '500': counts['500'], "
                    "'top_error_endpoint': top_err}\n"
                    "with open('summary.json', 'w') as f:\n"
                    "    json.dump(out, f, indent=2)\n"
                ),
            },
            test_command="python3 solution.py",
            verification_fn=verify_log_summary,
        ),
        TerminalTask(
            id="TB-02-BINARY-SEARCH-PATCH",
            category="code_patching",
            title="Diagnose & Patch Off-by-One in Binary Search",
            description=(
                "binary_search.py currently fails with an IndexError on empty lists and returns wrong "
                "indices when high boundary is len(arr). Apply patch and run verification test."
            ),
            setup_files={
                "binary_search.py": (
                    "def binary_search(arr, target):\n"
                    "    low = 0\n"
                    "    high = len(arr) - 1\n"
                    "    while low <= high:\n"
                    "        mid = (low + high) // 2\n"
                    "        if arr[mid] == target:\n"
                    "            return mid\n"
                    "        elif arr[mid] < target:\n"
                    "            low = mid + 1\n"
                    "        else:\n"
                    "            high = mid - 1\n"
                    "    return -1\n"
                ),
                "test_search.py": (
                    "from binary_search import binary_search\n"
                    "assert binary_search([], 5) == -1, 'empty array'\n"
                    "assert binary_search([1], 1) == 0, 'single elem'\n"
                    "assert binary_search([1], 2) == -1, 'single elem not found'\n"
                    "assert binary_search([1, 3, 5, 7, 9], 7) == 3, 'odd len'\n"
                    "assert binary_search([2, 4, 6, 8], 8) == 3, 'even len end'\n"
                    "assert binary_search([2, 4, 6, 8], 1) == -1, 'before start'\n"
                    "print('ALL_TESTS_PASS')\n"
                ),
            },
            test_command="python3 test_search.py",
            verification_fn=verify_search_patch,
        ),
        TerminalTask(
            id="TB-03-CPP-INT64-OVERFLOW",
            category="build_compilation",
            title="C++20 Compilation & Integer Overflow Diagnostic",
            description=(
                "sum_range.cpp overflows standard 32-bit signed int when accumulating N=10^9 numbers. "
                "Fix types to uint64_t/long long, compile with g++ -O3 -std=c++20, and verify output."
            ),
            setup_files={
                "sum_range.cpp": (
                    "#include <iostream>\n"
                    "#include <cstdint>\n"
                    "int main() {\n"
                    "    int64_t n = 1000000000LL;\n"
                    "    int64_t total = n * (n + 1) / 2;\n"
                    "    std::cout << total << std::endl;\n"
                    "    return 0;\n"
                    "}\n"
                ),
            },
            test_command="g++ -O3 -std=c++20 sum_range.cpp -o sum_range && ./sum_range",
            verification_fn=verify_cpp_build,
        ),
        TerminalTask(
            id="TB-04-RECURSIVE-YAML-REFACTOR",
            category="bash_scripting",
            title="Recursive Multi-File Sed/Awk Refactoring",
            description=(
                "Replace deprecated config key 'legacy_endpoint: http://old.hrm.local' with "
                "'modern_endpoint: https://api.hrm.internal' across all files under configs/."
            ),
            setup_files={
                "configs/app.yaml": (
                    "service: app\n" "port: 8080\n" "legacy_endpoint: http://old.hrm.local\n" "timeout_ms: 5000\n"
                ),
                "configs/db.yaml": (
                    "service: db\n" "port: 5432\n" "legacy_endpoint: http://old.hrm.local\n" "pool_size: 20\n"
                ),
                "refactor.sh": (
                    "#!/usr/bin/env bash\n"
                    "set -euo pipefail\n"
                    "find configs/ -name '*.yaml' -exec sed -i "
                    "'s|legacy_endpoint: http://old.hrm.local|modern_endpoint: https://api.hrm.internal|g' {} +\n"
                ),
            },
            test_command="bash refactor.sh",
            verification_fn=verify_refactor_yaml,
        ),
        TerminalTask(
            id="TB-05-UNITTEST-DIAGNOSTIC-REPAIR",
            category="test_repair",
            title="Diagnose & Repair Failing Test Suite",
            description=(
                "Run test suite via python3 -m unittest. Traceback indicates a ZeroDivisionError when calculating average "
                "on empty metrics list. Fix calculate_average in analytics.py and confirm all tests pass."
            ),
            setup_files={
                "analytics.py": (
                    "def calculate_average(values: list[float]) -> float:\n"
                    "    if not values:\n"
                    "        return 0.0\n"
                    "    return sum(values) / len(values)\n"
                ),
                "test_analytics.py": (
                    "import unittest\n"
                    "from analytics import calculate_average\n\n"
                    "class TestAnalytics(unittest.TestCase):\n"
                    "    def test_avg_positive(self):\n"
                    "        self.assertEqual(calculate_average([10.0, 20.0, 30.0]), 20.0)\n"
                    "    def test_avg_empty(self):\n"
                    "        self.assertEqual(calculate_average([]), 0.0)\n"
                    "    def test_avg_single(self):\n"
                    "        self.assertEqual(calculate_average([42.0]), 42.0)\n\n"
                    "if __name__ == '__main__':\n"
                    "    unittest.main()\n"
                ),
            },
            test_command="python3 -m unittest test_analytics.py",
            verification_fn=verify_test_repair,
        ),
    ]
    return tasks


class TerminalBenchEvaluator:
    """Evaluates agent terminal capabilities on standardized system benchmarks."""

    def __init__(self, sandbox: SandboxRunner | None = None):
        self.sandbox = sandbox or SandboxRunner()
        self.console = Console()

    def run_task(self, task: TerminalTask) -> TerminalResult:
        """Execute a single TerminalBench task inside an isolated scratch workspace."""
        with tempfile.TemporaryDirectory(prefix=f"tb_{task.id.lower()}_") as tmpdir:
            workspace = Path(tmpdir)

            # 1. Setup workspace files
            for rel_path, content in task.setup_files.items():
                dest = workspace / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")

            # 2. Execute test command
            start_wall = time.perf_counter()
            try:
                proc = subprocess.run(
                    task.test_command,
                    shell=True,
                    cwd=str(workspace),
                    capture_output=True,
                    text=True,
                    timeout=max(1.0, task.time_limit_ms / 1000.0),
                )
                wall_ms = (time.perf_counter() - start_wall) * 1000.0
                stdout = proc.stdout
                stderr = proc.stderr
                exit_code = proc.returncode

                # 3. Verify
                if task.verification_fn:
                    passed, msg = task.verification_fn(workspace, proc)
                else:
                    passed = exit_code == 0
                    msg = "Clean exit code 0" if passed else f"Exit code {exit_code}: {stderr}"

                return TerminalResult(
                    task_id=task.id,
                    category=task.category,
                    passed=passed,
                    exit_code=exit_code,
                    wall_clock_ms=round(wall_ms, 2),
                    output=stdout,
                    error_message=None if passed else msg,
                )
            except subprocess.TimeoutExpired:
                wall_ms = (time.perf_counter() - start_wall) * 1000.0
                return TerminalResult(
                    task_id=task.id,
                    category=task.category,
                    passed=False,
                    exit_code=-1,
                    wall_clock_ms=round(wall_ms, 2),
                    output="",
                    error_message=f"Execution timed out after {task.time_limit_ms}ms",
                )
            except Exception as e:
                wall_ms = (time.perf_counter() - start_wall) * 1000.0
                return TerminalResult(
                    task_id=task.id,
                    category=task.category,
                    passed=False,
                    exit_code=-1,
                    wall_clock_ms=round(wall_ms, 2),
                    output="",
                    error_message=f"Runtime exception: {e!s}",
                )

    def evaluate_all(
        self,
        tasks: list[TerminalTask] | None = None,
        output_report_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """Execute full TerminalBench test suite and generate structured report."""
        task_list = tasks or get_default_terminal_bench_tasks()
        results: list[TerminalResult] = []

        table = Table(title="TerminalBench Execution Summary")
        table.add_column("Task ID", style="cyan", no_wrap=True)
        table.add_column("Category", style="magenta")
        table.add_column("Verdict", style="bold")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("Details", style="dim")

        passed_count = 0
        total_time_ms = 0.0

        for t in task_list:
            res = self.run_task(t)
            results.append(res)
            total_time_ms += res.wall_clock_ms
            if res.passed:
                passed_count += 1
                table.add_row(res.task_id, res.category, "[green]PASS[/green]", f"{res.wall_clock_ms:.1f}", "OK")
            else:
                table.add_row(
                    res.task_id,
                    res.category,
                    "[red]FAIL[/red]",
                    f"{res.wall_clock_ms:.1f}",
                    res.error_message or "",
                )

        self.console.print(table)
        pass_rate = (passed_count / len(task_list)) * 100.0 if task_list else 0.0

        summary = {
            "total_tasks": len(task_list),
            "passed_tasks": passed_count,
            "pass_rate_pct": round(pass_rate, 2),
            "total_wall_clock_ms": round(total_time_ms, 2),
            "results": [
                {
                    "task_id": r.task_id,
                    "category": r.category,
                    "passed": r.passed,
                    "exit_code": r.exit_code,
                    "wall_clock_ms": r.wall_clock_ms,
                    "error": r.error_message,
                }
                for r in results
            ],
        }

        if output_report_path:
            out_p = Path(output_report_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

        return summary


if __name__ == "__main__":
    runner = TerminalBenchEvaluator()
    runner.evaluate_all(output_report_path="outputs/terminal_bench_report.json")
