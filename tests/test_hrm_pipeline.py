"""Comprehensive end-to-end pytest suite for HRM competitive programming agent.

Tests:
1. GBNF grammars compliance and loading in llama_cpp
2. Unified diff patcher (strict line-offset matching, fuzzy fallback, syntax validation)
3. Sandbox verdicts: CE, AC, WA, TLE, MLE, RE and os.wait4 telemetry extraction
4. Fuzz oracle & worst-case input generation, differential consensus, float checkers
5. Jev critic escalation state machine (Node 1 -> Node 2 -> Node 3 -> Node 1 escalation -> Node 2 -> Node 3 PASS)
6. Stratified benchmark evaluator & 3-way ablation metrics across 5 synthetic problems
"""

import json
import os
from pathlib import Path

import pytest

from hrm_agent.hrm_loop import HRMAgentLoop
from hrm_agent.llm_engine import MockLLMEngine
from hrm_agent.patcher import apply_patch, validate_syntax_balance
from hrm_agent.schemas import (
    JevVerdict,
    PlannerBlueprint,
    ProblemSpec,
)
from judge.benchmark_eval import BenchmarkEvaluator
from judge.fuzz_oracle import (
    FloatToleranceChecker,
    FuzzCaseGenerator,
)
from judge.sandbox import SandboxRunner


# =====================================================================
# Test 1: GBNF Grammar Validity
# =====================================================================
def test_gbnf_grammars_load():
    """Verify that all 5 GBNF grammar files compile without error in llama_cpp."""
    llama_cpp = pytest.importorskip("llama_cpp")
    grammar_dir = Path("grammars")
    grammar_files = [
        "planner.gbnf",
        "cpp_raw.gbnf",
        "py_raw.gbnf",
        "unified_diff.gbnf",
        "critic.gbnf",
    ]

    for g_file in grammar_files:
        path = grammar_dir / g_file
        assert path.exists(), f"Grammar file missing: {path}"
        content = path.read_text(encoding="utf-8")
        # Compile grammar using llama_cpp
        compiled = llama_cpp.LlamaGrammar.from_string(content)
        assert compiled is not None, f"Failed to compile grammar {g_file}"


def test_schema_validations():
    """Verify strict Pydantic v2 schemas for blueprints and verdicts."""
    bp_data = {
        "analysis": "Two pointer window maintaining count of distinct elements",
        "paradigm": "two_pointers",
        "data_structures": ["vector", "map"],
        "time_complexity": "O(N)",
        "space_complexity": "O(N)",
        "use_int64": True,
        "edge_cases": ["N=1", "all distinct", "all identical"],
        "steps": [
            "Initialize two pointers at index 0",
            "Expand right pointer adding elements to frequency map",
            "Shrink left pointer while window remains valid",
        ],
    }
    bp = PlannerBlueprint.model_validate(bp_data)
    assert bp.paradigm == "two_pointers"
    assert bp.use_int64 is True

    verdict_data = {
        "analysis": "Constant factor too high inside nested loop, causing TLE",
        "status": "OPTIMIZE",
        "bottleneck_class": "constant_factor",
        "escalate_to_planner": False,
        "target_lines": "L22-L35",
        "patch_directive": "Replace std::endl with '\\n' and hoist vector allocation",
    }
    verdict = JevVerdict.model_validate(verdict_data)
    assert verdict.status == "OPTIMIZE"
    assert verdict.escalate_to_planner is False


# =====================================================================
# Test 2: Unified Diff Patcher
# =====================================================================
def test_patcher_strict_match():
    """Verify strict line-offset patch application."""
    source_code = (
        "#include <iostream>\n"
        "using namespace std;\n"
        "\n"
        "int main() {\n"
        "    int a, b;\n"
        "    cin >> a >> b;\n"
        "    cout << a - b << '\\n';\n"
        "    return 0;\n"
        "}\n"
    )

    diff = (
        "--- a/solution.cpp\n"
        "+++ b/solution.cpp\n"
        "@@ -6,3 +6,3 @@\n"
        "     cin >> a >> b;\n"
        "-    cout << a - b << '\\n';\n"
        "+    cout << a + b << '\\n';\n"
        "     return 0;\n"
    )

    ok, patched, meta = apply_patch(source_code, diff, language="cpp")
    assert ok is True
    assert meta.applied_cleanly is True
    assert "cout << a + b" in patched
    assert "cout << a - b" not in patched


def test_patcher_fuzzy_fallback():
    """Verify fuzzy hunk fallback when line numbers are shifted."""
    source_code = (
        "// Additional preamble comment line 1\n"
        "// Additional preamble comment line 2\n"
        "// Additional preamble comment line 3\n"
        "#include <iostream>\n"
        "using namespace std;\n"
        "\n"
        "int main() {\n"
        "    int a, b;\n"
        "    cin >> a >> b;\n"
        "    cout << a - b << '\\n';\n"
        "    return 0;\n"
        "}\n"
    )

    # Offset is intentionally inaccurate (points to line 2 instead of line 9)
    diff = (
        "--- a/solution.cpp\n"
        "+++ b/solution.cpp\n"
        "@@ -2,3 +2,3 @@\n"
        "     cin >> a >> b;\n"
        "-    cout << a - b << '\\n';\n"
        "+    cout << a * b << '\\n';\n"
        "     return 0;\n"
    )

    ok, patched, meta = apply_patch(source_code, diff, language="cpp")
    assert ok is True
    assert meta.applied_cleanly is False  # Applied via fuzzy fallback
    assert "cout << a * b" in patched


def test_syntax_balance_checker():
    """Verify brace and bracket balance checking."""
    balanced_code = "int main() { if (x > 0) { return 1; } return 0; }"
    ok, err = validate_syntax_balance(balanced_code, language="cpp")
    assert ok is True
    assert err is None

    unbalanced_code = "int main() { if (x > 0) { return 1; return 0; }"
    ok, err = validate_syntax_balance(unbalanced_code, language="cpp")
    assert ok is False
    assert "Unclosed '{'" in err


# =====================================================================
# Test 3: Sandbox Execution & Telemetry Extraction
# =====================================================================
def test_sandbox_verdict_ac():
    """Verify AC verdict on correct C++ program."""
    sandbox = SandboxRunner()
    code = (
        "#include <iostream>\n"
        "using namespace std;\n"
        "int main() {\n"
        "    int a, b;\n"
        "    if (cin >> a >> b) {\n"
        "        cout << a + b << '\\n';\n"
        "    }\n"
        "    return 0;\n"
        "}\n"
    )
    telem = sandbox.run_code(
        code=code,
        language="cpp",
        input_data="10 25\n",
        expected_output="35\n",
        time_limit_ms=2000,
        memory_limit_mb=256,
    )
    assert telem.verdict == "AC"
    assert telem.exit_code == 0
    assert telem.stdout.strip() == "35"
    assert telem.wall_clock_ms > 0.0


def test_sandbox_verdict_wa():
    """Verify WA verdict when program produces wrong answer."""
    sandbox = SandboxRunner()
    code = (
        "#include <iostream>\n"
        "using namespace std;\n"
        "int main() {\n"
        "    int a, b; cin >> a >> b;\n"
        "    cout << a - b << '\\n';\n"
        "    return 0;\n"
        "}\n"
    )
    telem = sandbox.run_code(
        code=code,
        language="cpp",
        input_data="10 25\n",
        expected_output="35\n",
        time_limit_ms=2000,
        memory_limit_mb=256,
    )
    assert telem.verdict == "WA"
    assert telem.exit_code == 0


def test_sandbox_verdict_ce():
    """Verify CE verdict on invalid C++ code."""
    sandbox = SandboxRunner()
    bad_code = "int main() { invalid_syntax_error__ }"
    telem = sandbox.run_code(
        code=bad_code,
        language="cpp",
        input_data="1\n",
        expected_output="1\n",
    )
    assert telem.verdict == "CE"
    assert telem.exit_code != 0
    assert "error" in telem.stderr.lower()


def test_sandbox_verdict_tle():
    """Verify TLE verdict on an infinite loop."""
    sandbox = SandboxRunner()
    loop_code = (
        "#include <iostream>\nint main() {\n    volatile int x = 0;\n    while (true) { x++; }\n    return 0;\n}\n"
    )
    telem = sandbox.run_code(
        code=loop_code,
        language="cpp",
        input_data="1\n",
        expected_output="1\n",
        time_limit_ms=300,  # 300 ms limit
    )
    assert telem.verdict == "TLE"


def test_sandbox_verdict_re():
    """Verify RE verdict on division by zero or abort."""
    sandbox = SandboxRunner()
    crash_code = "#include <iostream>\n#include <cstdlib>\nint main() {\n    abort();\n    return 0;\n}\n"
    telem = sandbox.run_code(
        code=crash_code,
        language="cpp",
        input_data="1\n",
        expected_output="1\n",
        time_limit_ms=2000,
    )
    assert telem.verdict == "RE"


def test_sandbox_verdict_mle():
    """Verify MLE detection when process attempts to allocate memory beyond limit."""
    sandbox = SandboxRunner()
    mle_code = (
        "#include <iostream>\n"
        "#include <vector>\n"
        "int main() {\n"
        "    try {\n"
        "        std::vector<char> v(500 * 1024 * 1024, 1); // 500 MB\n"
        "        std::cout << v[0] << '\\n';\n"
        "    } catch (...) {\n"
        "        return 1;\n"
        "    }\n"
        "    return 0;\n"
        "}\n"
    )
    telem = sandbox.run_code(
        code=mle_code,
        language="cpp",
        input_data="1\n",
        expected_output="1\n",
        time_limit_ms=2000,
        memory_limit_mb=64,  # Strict 64 MB limit
    )
    # Under setrlimit/prlimit, allocating 500MB throws bad_alloc or segfaults/MLE
    assert telem.verdict in ("MLE", "RE")


# =====================================================================
# Test 4: Fuzz Oracle & Differential Consensus
# =====================================================================
def test_fuzz_case_generator():
    """Verify property-based and worst-case input generation."""
    spec = ProblemSpec(
        id="CF-9999Z",
        title="Array Peak",
        description="Find peak element in array",
        constraints="1 <= N <= 2*10^5, 1 <= A[i] <= 10^9",
    )

    boundaries = FuzzCaseGenerator.generate_boundary_cases(spec)
    assert len(boundaries) >= 5
    assert any("0" in b for b in boundaries)

    worst_case = FuzzCaseGenerator.generate_worst_case(spec)
    assert len(worst_case.splitlines()) >= 2
    # Check that it generated stress array
    assert int(worst_case.splitlines()[1]) >= 10000

    star_graph = FuzzCaseGenerator.generate_graph_cases(num_nodes=50, graph_type="star")
    assert "1 2" in star_graph
    assert "1 50" in star_graph


def test_float_tolerance_checker():
    """Verify custom float checker with 1e-6 tolerance."""
    checker = FloatToleranceChecker(tolerance=1e-6)
    assert checker("3.14159265", "3.14159260") is True
    assert checker("3.14159265", "3.15159265") is False
    assert checker("1.0 2.0", "1.0 2.0") is True


# =====================================================================
# Test 5: Jev Critic Escalation State Machine
# =====================================================================
def test_jev_critic_escalation_loop():
    """
    Test full state machine with Jev escalation:
    1. Node 1 emits an initial O(N^2) blueprint
    2. Node 2 produces working code for samples
    3. Node 3 Critic flags asymptotic bottleneck and sets escalate_to_planner = true
    4. Loop escalates back to Node 1 with critique context
    5. Node 1 emits O(N) blueprint
    6. Node 2 produces linear code
    7. Node 3 emits PASS
    """
    # Program 1: Quadratic implementation
    quad_code = (
        "#include <iostream>\n"
        "#include <vector>\n"
        "using namespace std;\n"
        "int main() {\n"
        "    int n; if (!(cin >> n)) return 0;\n"
        "    vector<int> a(n);\n"
        "    for (int i = 0; i < n; ++i) cin >> a[i];\n"
        "    long long sum = 0;\n"
        "    for (int i = 0; i < n; ++i) sum += a[i];\n"
        "    cout << sum << '\\n';\n"
        "    return 0;\n"
        "}\n"
    )

    # Responses sequence for MockLLM
    # Call 1 (Planner): Initial blueprint
    bp_1 = json.dumps(
        {
            "analysis": "Quadratic brute force comparison across pairs",
            "paradigm": "brute_force_backtrack",
            "data_structures": ["vector"],
            "time_complexity": "O(N^2)",
            "space_complexity": "O(N)",
            "use_int64": True,
            "edge_cases": ["N=1"],
            "steps": ["Scan all pairs"],
        }
    )
    # Call 2 (Coder): Plan to code
    c_1 = quad_code
    # Call 3 (Critic): Verdict with escalate_to_planner = True
    crit_1 = json.dumps(
        {
            "analysis": "Asymptotic O(N^2) time complexity will TLE for N=2*10^5",
            "status": "OPTIMIZE",
            "bottleneck_class": "asymptotic_time",
            "escalate_to_planner": True,
            "target_lines": "L8-L10",
            "patch_directive": "Replan with linear O(N) prefix sum or hash set",
        }
    )
    # Call 4 (Planner after escalation): Linear blueprint
    bp_2 = json.dumps(
        {
            "analysis": "Linear O(N) traversal maintaining running sum",
            "paradigm": "greedy",
            "data_structures": ["vector"],
            "time_complexity": "O(N)",
            "space_complexity": "O(1)",
            "use_int64": True,
            "edge_cases": ["N=1", "overflow"],
            "steps": ["Single pass linear summation"],
        }
    )
    # Call 5 (Coder): Optimal code
    c_2 = quad_code
    # Call 6 (Critic): PASS verdict
    crit_2 = json.dumps(
        {
            "analysis": "Optimal O(N) linear time, executes well within time budget",
            "status": "PASS",
            "bottleneck_class": "none",
            "escalate_to_planner": False,
            "target_lines": "none",
            "patch_directive": "None",
        }
    )

    planner_mock = MockLLMEngine([bp_1, bp_2])
    coder_mock = MockLLMEngine([c_1, c_2])
    critic_mock = MockLLMEngine([crit_1, crit_2])

    spec = ProblemSpec(
        id="CHEF-SUM01",
        title="Sum of Array",
        description="Compute sum of N numbers",
        constraints="1 <= N <= 2*10^5",
        sample_inputs=["3\n1 2 3\n"],
        sample_outputs=["6\n"],
    )

    loop = HRMAgentLoop(
        node1_planner=planner_mock,
        node2_coder=coder_mock,
        node3_critic=critic_mock,
        n_plans=1,
        n_candidates=1,
        max_inner_iters=1,
        max_critic_iters=2,
        max_planner_escalations=1,
    )

    trace = loop.solve(spec)

    assert trace.solved is True
    assert trace.final_verdict == "AC"
    assert trace.macro_steps == 2  # Proves replanning escalation executed!
    assert len(trace.critic_verdicts) == 2
    assert trace.critic_verdicts[0].escalate_to_planner is True
    assert trace.critic_verdicts[1].status == "PASS"


# =====================================================================
# Test 6: Stratified Benchmark Evaluator & 3-way Ablation across 5 Problems
# =====================================================================
def test_benchmark_ablation_5_synthetic_problems(tmp_path):
    """
    Verify benchmark evaluator running 3-way ablation across 5 synthetic problems
    spanning rating buckets (<1200, 1200-1500, 1500-1800, 1800-2100, 2100+).
    """
    problems = [
        ProblemSpec(
            id="PROB-1",
            title="Array Sum",
            description="Sum of N numbers",
            constraints="1 <= N <= 1000",
            rating=950,
            sample_inputs=["1\n3\n1 2 3\n"],
            sample_outputs=["6\n"],
        ),
        ProblemSpec(
            id="PROB-2",
            title="Two Pointers Subarray",
            description="Find subarray",
            constraints="1 <= N <= 10^5",
            rating=1300,
            sample_inputs=["1\n3\n1 2 3\n"],
            sample_outputs=["6\n"],
        ),
        ProblemSpec(
            id="PROB-3",
            title="Tree Max Path",
            description="Diameter of tree",
            constraints="1 <= N <= 10^5",
            rating=1600,
            sample_inputs=["1\n3\n1 2 3\n"],
            sample_outputs=["6\n"],
        ),
        ProblemSpec(
            id="PROB-4",
            title="Segment Tree Queries",
            description="Point update range sum",
            constraints="1 <= N <= 2*10^5",
            rating=1900,
            sample_inputs=["1\n3\n1 2 3\n"],
            sample_outputs=["6\n"],
        ),
        ProblemSpec(
            id="PROB-5",
            title="Graph Articulation Points",
            description="Find cut vertices",
            constraints="1 <= N <= 2*10^5",
            rating=2200,
            sample_inputs=["1\n3\n1 2 3\n"],
            sample_outputs=["6\n"],
        ),
    ]

    report_path = str(tmp_path / "eval_report.json")
    evaluator = BenchmarkEvaluator(
        node1_planner=MockLLMEngine(),
        node2_coder=MockLLMEngine(),
        node3_critic=MockLLMEngine(),
    )

    results = evaluator.evaluate_ablation(problems, output_report_path=report_path)

    assert "pass_at_1" in results
    assert "node2_only" in results["pass_at_1"]
    assert "node1_2" in results["pass_at_1"]
    assert "full_hrm" in results["pass_at_1"]
    assert results["full_hrm"]["solved"] == 5
    assert results["pass_at_1"]["full_hrm"] == 100.0

    # Verify bucket stats
    assert len(results["bucket_stats"]) == 5
    assert results["bucket_stats"]["<1200"]["total"] == 1
    assert results["bucket_stats"]["1200-1500"]["total"] == 1
    assert results["bucket_stats"]["1500-1800"]["total"] == 1
    assert results["bucket_stats"]["1800-2100"]["total"] == 1
    assert results["bucket_stats"]["2100+"]["total"] == 1

    # Verify telemetry and Jev metrics
    assert "jev_precision" in results
    assert "jev_false_positive_rate" in results
    assert results["avg_tokens_per_problem"] > 0
    assert os.path.exists(report_path)
