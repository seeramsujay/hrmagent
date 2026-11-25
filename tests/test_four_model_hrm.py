"""Supervisor/Coder/Verifier pipeline tests. Engines are scripted: these test plumbing, not ability."""

from hrm_agent.four_model_agent import FourModelAgent
from hrm_agent.llm_engine import MockLLMEngine
from hrm_agent.schemas import ProblemSpec

SPEC = ProblemSpec(
    id="TEST-DOUBLE",
    title="Double",
    description="Print twice the integer.",
    sample_inputs=["21\n"],
    sample_outputs=["42\n"],
)


def _agent(tmp_path, code, patch=""):
    return FourModelAgent(
        supervisor_engine=MockLLMEngine(["not json"]),
        coder_engine=MockLLMEngine([code]),
        verifier_engine=MockLLMEngine([patch, patch]),
        workspace_dir=tmp_path,
    )


def test_verdict_comes_from_the_engines_code(tmp_path):
    trace = _agent(tmp_path, "```python\nprint(int(input()) * 2)\n```").solve(SPEC)
    assert trace.solved and trace.final_verdict == "AC"
    assert trace.blueprint is None  # unparseable plans are not invented


def test_no_canned_solution_replaces_wrong_code(tmp_path):
    trace = _agent(tmp_path, "print(0)\n").solve(SPEC)
    assert not trace.solved and trace.final_verdict == "WA"
    assert trace.synthesized_code == "print(0)\n"


def test_verifier_patch_repairs_code(tmp_path):
    patch = "<<<<<<< SEARCH\nprint(0)\n=======\nprint(int(input()) * 2)\n>>>>>>> REPLACE"
    trace = _agent(tmp_path, "print(0)\n", patch).solve(SPEC)
    assert trace.solved and trace.repair_iterations == 1
