"""Test suite for TerminalBench evaluation framework."""

import json

from judge.terminal_bench import (
    TerminalBenchEvaluator,
    get_default_terminal_bench_tasks,
)


def test_default_terminal_bench_tasks_structure():
    """Verify standard TerminalBench task definitions."""
    tasks = get_default_terminal_bench_tasks()
    assert len(tasks) >= 5

    task_ids = [t.id for t in tasks]
    assert "TB-01-LOG-METRICS" in task_ids
    assert "TB-02-BINARY-SEARCH-PATCH" in task_ids
    assert "TB-03-CPP-INT64-OVERFLOW" in task_ids
    assert "TB-04-RECURSIVE-YAML-REFACTOR" in task_ids
    assert "TB-05-UNITTEST-DIAGNOSTIC-REPAIR" in task_ids

    for t in tasks:
        assert t.category in [
            "log_diagnostics",
            "code_patching",
            "build_compilation",
            "bash_scripting",
            "test_repair",
            "environment_inspection",
        ]
        assert len(t.setup_files) > 0
        assert t.test_command != ""


def test_terminal_bench_execution_and_reporting(tmp_path):
    """Execute all TerminalBench tasks and verify 100% pass rate and telemetry."""
    evaluator = TerminalBenchEvaluator()
    report_file = tmp_path / "terminal_bench_report.json"

    summary = evaluator.evaluate_all(output_report_path=str(report_file))

    assert summary["total_tasks"] >= 5
    tot = summary["total_tasks"]
    passed = summary["passed_tasks"]
    assert passed == tot, f"Expected all tasks to pass, got: {passed}/{tot}"
    assert summary["pass_rate_pct"] == 100.0
    assert summary["total_wall_clock_ms"] > 0

    assert report_file.exists()
    saved = json.loads(report_file.read_text(encoding="utf-8"))
    assert saved["pass_rate_pct"] == 100.0
    assert len(saved["results"]) == summary["total_tasks"]
