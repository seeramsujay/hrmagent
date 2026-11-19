"""Test suite for CodeChef dataset curation and HRM Dual-Pair training pipeline."""

import json

from hrm_agent.patcher import apply_patch
from hrm_agent.schemas import JevVerdict, PlannerBlueprint
from training.dataset_miner import DatasetMiner


def test_codechef_curation_and_trajectory_mining(tmp_path):
    """Verify CodeChef challenge ingestion, DuckDB relational schema, and trajectory mining."""
    miner = DatasetMiner(output_dir=str(tmp_path))
    miner.populate_synthetic_seed()

    # Ingest CodeChef challenges from local cached corpus
    cc_count = miner.populate_codechef_challenges(
        json_path="data/codechef_raw.json",
        limit=12,
        streaming=False,
    )
    assert cc_count >= 5, f"Expected at least 5 CodeChef challenges, got {cc_count}"

    # Verify DuckDB problems table contains CodeChef rows
    cc_rows = miner.con.execute("SELECT problem_id, title, source FROM problems WHERE source = 'CODECHEF'").fetchall()
    assert len(cc_rows) >= 5
    assert all(r[2] == "CODECHEF" for r in cc_rows)

    # Verify trajectories are mined (buggy failure -> AC fix)
    trajectories = miner.mine_trajectories()
    assert len(trajectories) >= 2, "Expected at least 2 mined trajectory pairs"
    for traj in trajectories:
        assert "diff" in traj
        assert traj["jaccard"] >= 0.50
        assert "@@" in traj["diff"]


def test_dual_pair_balanced_datasets(tmp_path):
    """Verify that build_balanced_datasets outputs valid training pairs for both roles."""
    miner = DatasetMiner(output_dir=str(tmp_path))
    miner.populate_synthetic_seed()
    miner.populate_codechef_challenges(json_path="data/codechef_raw.json", limit=10)

    paths = miner.build_balanced_datasets()
    assert "node1_blueprints.jsonl" in paths
    assert "node2_plan_to_code.jsonl" in paths
    assert "node2_error_to_diff.jsonl" in paths
    assert "node2_dpo.jsonl" in paths
    assert "node3_critic.jsonl" in paths

    # 1. Verify Role A: Planner / Architect blueprints
    with open(paths["node1_blueprints.jsonl"]) as f:
        node1_lines = [json.loads(line) for line in f]
    assert len(node1_lines) >= 5
    for record in node1_lines:
        assert "prompt" in record and "completion" in record
        bp_dict = json.loads(record["completion"])
        bp = PlannerBlueprint.model_validate(bp_dict)
        assert bp.paradigm in [
            "greedy",
            "dynamic_programming",
            "two_pointers",
            "binary_search",
            "graph_bfs_dfs",
            "shortest_path",
            "disjoint_set_union",
            "segment_tree_fenwick",
            "math_number_theory",
            "constructive",
            "bitmask",
            "divide_and_conquer",
            "flow_matching",
            "brute_force_backtrack",
            "game_theory",
        ]
        assert bp.time_complexity.startswith("O(")

    # 2. Verify Role B: Coder plan-to-code implementations
    with open(paths["node2_plan_to_code.jsonl"]) as f:
        node2_lines = [json.loads(line) for line in f]
    assert len(node2_lines) >= 5
    for record in node2_lines:
        assert "Blueprint:" in record["prompt"]
        code = record["completion"]
        assert len(code) > 10

    # 3. Verify Role A: Critic & Hallucination/Bug detection
    with open(paths["node3_critic.jsonl"]) as f:
        node3_lines = [json.loads(line) for line in f]
    assert len(node3_lines) >= 2
    for record in node3_lines:
        verdict_dict = json.loads(record["completion"])
        verdict = JevVerdict.model_validate(verdict_dict)
        assert verdict.status in ["PASS", "BUG", "OPTIMIZE"]
        assert verdict.bottleneck_class in [
            "none",
            "asymptotic_time",
            "constant_factor",
            "memory_limit",
            "integer_overflow",
            "edge_case_miss",
            "logic_flaw",
            "io_overhead",
        ]


def test_dual_pair_patch_application():
    """Verify that a surgical patch generated from critic directive applies cleanly."""
    original_code = (
        "def solve(n, k):\n    ans = 0\n    for i in range(1, len(k)):\n        ans += k[i]\n    return ans\n"
    )
    diff = (
        "--- a/solution.py\n"
        "+++ b/solution.py\n"
        "@@ -3,3 +3,3 @@\n"
        "-    for i in range(1, len(k)):\n"
        "+    for i in range(len(k)):\n"
    )
    success, patched_code, meta = apply_patch(original_code, diff, language="python")
    assert success
    assert "for i in range(len(k)):" in patched_code
    assert meta.applied_cleanly
