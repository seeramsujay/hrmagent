"""Test suite for the Four-Model Hierarchical Reasoning Architecture (HRM + Supervisor + Coder + Verifier)."""

from hrm_agent.four_model_agent import FourModelAgent
from hrm_agent.hrm_latent_engine import HRMLatentConfig, HRMLatentReasoner, HRMLatentSolver
from hrm_agent.schemas import ProblemSpec


def test_hrm_latent_recurrent_convergence():
    """Verify Sapient-style dual-timescale recurrence converges and computes halting probability."""
    config = HRMLatentConfig(hidden_size=128, num_heads=2, H_cycles=2, L_cycles=3, max_steps=6)
    reasoner = HRMLatentReasoner(config)

    import torch

    tokens = torch.tensor([[10, 20, 30, 40, 50]])
    concept_repr, steps, halt_prob = reasoner.reason_to_convergence(tokens)

    assert steps >= 1
    assert concept_repr.shape == (1, 5, 128)
    assert 0.0 <= halt_prob <= 1.0


def test_hrm_latent_solver_invariants():
    """Verify HRMLatentSolver extracts invariant and paradigm without generating text CoT tokens."""
    solver = HRMLatentSolver(device="cpu")

    # 1. Game theory problem
    gt_res = solver.discover_invariant(
        problem_description="Nim stone game with multiple quarries and alternating turns to win or lose.",
        constraints="1 <= N <= 10^5",
    )
    assert gt_res["paradigm"] == "game_theory"
    assert "Sprague-Grundy" in gt_res["invariant"]
    assert gt_res["latent_steps"] >= 1

    # 2. Dynamic programming problem
    dp_res = solver.discover_invariant(
        problem_description="Calculate the maximum points possible using dynamic programming.",
        constraints="1 <= N <= 10^5",
    )
    assert dp_res["paradigm"] == "dynamic_programming"
    assert "Optimal substructure" in dp_res["invariant"]


def test_four_model_agent_closed_loop_solve(tmp_path):
    """Verify complete 4-tier closed-loop solving on CodeChef challenge."""
    agent = FourModelAgent(workspace_dir=tmp_path, device="cpu")

    spec = ProblemSpec(
        id="TEST-CC-FLOW017",
        title="Second Largest",
        description="Three numbers A, B and C are given. Find the second largest among them.",
        constraints="1 <= T <= 100, 1 <= A, B, C <= 10^6",
        sample_inputs=["2\n10 20 30\n500 100 250\n"],
        sample_outputs=["20\n250\n"],
    )

    trace = agent.solve(spec, language="python")

    assert trace.problem_id == "TEST-CC-FLOW017"
    assert trace.solved is True
    assert trace.final_verdict == "AC"
    assert "latent_steps" in trace.latent_invariant
    assert trace.blueprint is not None
    assert trace.verification_telemetry is not None
    assert trace.verification_telemetry.verdict == "AC"
    assert "def solve():" in trace.synthesized_code


def test_four_model_agent_game_theory_solve(tmp_path):
    """Verify 4-tier pipeline on Game Theory parity challenge."""
    agent = FourModelAgent(workspace_dir=tmp_path, device="cpu")

    spec = ProblemSpec(
        id="TEST-CHEF-NUMGAME",
        title="Chef Number Game",
        description="Alice and Bob play on integer N subtracting proper divisors. Parity game.",
        constraints="1 <= T <= 10, 1 <= N <= 10^9",
        sample_inputs=["2\n1\n2\n"],
        sample_outputs=["BOB\nALICE\n"],
    )

    trace = agent.solve(spec, language="python")

    assert trace.problem_id == "TEST-CHEF-NUMGAME"
    assert trace.solved is True
    assert trace.final_verdict == "AC"
    assert trace.latent_invariant["paradigm"] == "game_theory"
    assert "ALICE" in trace.synthesized_code
    assert "BOB" in trace.synthesized_code


def test_four_model_vs_baseline_comparison(tmp_path):
    """Verify side-by-side comparative benchmark execution between Baseline and HRM-4M."""
    from judge.compare_4m_vs_baseline import run_comparison

    out_file = tmp_path / "comp_report.json"
    summary = run_comparison(output_path=str(out_file))

    assert summary["total_problems"] == 10
    assert summary["hrm_4m_ac_count"] == 10
    assert summary["hrm_4m_pass_rate_pct"] == 100.0
    assert out_file.exists()
