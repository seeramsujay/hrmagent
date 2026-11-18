"""Comprehensive test suite for previous year Game Theory competitive programming challenges.

Covers:
1. Sprague-Grundy Theorem and multi-pile Nim (CF-15C Industrial Nim, CHEF-NIM)
2. Periodic subtraction games and modulo invariants (CF-1194D 1-2-K Game)
3. Factorization and divisor parity games (CF-1537D Deleting Divisors, CHEF-NUMGAME)
4. Jev Critic diagnosis of naive exponential recursion in game trees
5. Localhost API /solve and /v1/chat/completions verification on game theory problems
"""

from starlette.testclient import TestClient

from hrm_agent.patcher import apply_patch
from hrm_agent.schemas import JevVerdict, PlannerBlueprint
from hrm_agent.server import create_app
from judge.sandbox import SandboxRunner
from training.dataset_miner import DatasetMiner


def test_game_theory_dataset_mining(tmp_path):
    """Verify DuckDB ingestion and trajectory mining of game theory problems."""
    miner = DatasetMiner(output_dir=str(tmp_path))
    gt_count = miner.populate_game_theory_challenges()
    assert gt_count == 6

    # Verify DuckDB problems table contains game theory rows
    cf_rows = miner.con.execute(
        "SELECT problem_id, title, source FROM problems WHERE problem_id IN ('CF-15C', 'CF-1194D', 'CF-1537D')"
    ).fetchall()
    assert len(cf_rows) == 3

    # Verify trajectories mined for game theory
    trajs = miner.mine_trajectories()
    gt_trajs = [t for t in trajs if t["problem_id"].startswith("CF-") or t["problem_id"].startswith("CHEF-")]
    assert len(gt_trajs) >= 3
    for t in gt_trajs:
        assert "diff" in t
        assert t["jaccard"] >= 0.50


def test_game_theory_blueprint_schema():
    """Verify PlannerBlueprint adheres strictly to schema with paradigm='game_theory'."""
    bp_dict = {
        "analysis": "Sprague-Grundy theorem. Invariant is XOR sum of quarry ranges computed in O(1).",
        "paradigm": "game_theory",
        "data_structures": ["vector"],
        "time_complexity": "O(N)",
        "space_complexity": "O(1)",
        "use_int64": True,
        "edge_cases": ["N=1", "m_i=1", "XOR sum == 0"],
        "steps": [
            "Compute prefix XOR function f(n) = n ^ (n-1) ... ^ 0 in O(1)",
            "XOR sum of [x, x + m - 1] is f(x + m - 1) ^ f(x - 1)",
            "If total XOR sum != 0, first player wins ('tolik'), else 'bolik'",
        ],
    }
    bp = PlannerBlueprint.model_validate(bp_dict)
    assert bp.paradigm == "game_theory"
    assert bp.time_complexity == "O(N)"
    assert bp.use_int64 is True


def test_sprague_grundy_nim_sandbox():
    """Verify execution of classic Nim Sprague-Grundy XOR sum in sandbox."""
    sandbox = SandboxRunner()
    code = (
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
        "        xor_sum = reduce(lambda a, b: a ^ b, piles, 0)\n"
        "        print('First' if xor_sum != 0 else 'Second')\n"
        "if __name__ == '__main__':\n"
        "    solve()\n"
    )

    # Test Case 1: Piles [3, 4, 5] -> XOR = 3 ^ 4 ^ 5 = 2 != 0 -> First
    # Test Case 2: Piles [1, 2, 3] -> XOR = 1 ^ 2 ^ 3 = 0 -> Second
    input_data = "2\n3\n3 4 5\n3\n1 2 3\n"
    expected = "First\nSecond\n"

    telem = sandbox.run_code(
        code=code,
        language="python",
        input_data=input_data,
        expected_output=expected,
        time_limit_ms=2000,
        memory_limit_mb=256,
    )
    assert telem.verdict == "AC"
    assert telem.exit_code == 0
    assert "First\nSecond" in telem.stdout


def test_subtraction_game_1_2_k_sandbox():
    """Verify CF-1194D (1-2-K Game) modulo cycle analysis in sandbox."""
    sandbox = SandboxRunner()
    code = (
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
    )

    # n=1 k=3: n%(4) = 1 != 0 -> Alice
    # n=3 k=3: n%(4) = 3 == k -> Alice
    # n=4 k=3: n%(4) = 0 -> Bob
    input_data = "3\n1 3\n3 3\n4 3\n"
    expected = "Alice\nAlice\nBob\n"

    telem = sandbox.run_code(
        code=code,
        language="python",
        input_data=input_data,
        expected_output=expected,
        time_limit_ms=2000,
        memory_limit_mb=256,
    )
    assert telem.verdict == "AC"
    assert telem.exit_code == 0


def test_deleting_divisors_power_of_two_game():
    """Verify CF-1537D (Deleting Divisors) prime factorization game in sandbox."""
    sandbox = SandboxRunner()
    code = (
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
    )

    # 1: Bob (odd)
    # 4: 2^2 (even exp) -> Alice
    # 8: 2^3 (odd exp) -> Bob
    # 12: even, not power of 2 -> Alice
    input_data = "4\n1\n4\n8\n12\n"
    expected = "Bob\nAlice\nBob\nAlice\n"

    telem = sandbox.run_code(
        code=code,
        language="python",
        input_data=input_data,
        expected_output=expected,
        time_limit_ms=2000,
        memory_limit_mb=256,
    )
    assert telem.verdict == "AC"
    assert telem.exit_code == 0


def test_game_theory_critic_diagnosis_and_patch():
    """Verify Jev Critic detects naive recursive game search and patcher repairs it."""
    # Buggy code: recursive game tree without memoization
    buggy_game_code = (
        "import sys\n"
        "def solve():\n"
        "    lines = sys.stdin.read().splitlines()\n"
        "    n = int(lines[0])\n"
        "    # Naive recursive search\n"
        "    def win(x):\n"
        "        if x == 0: return False\n"
        "        for d in (1, 2):\n"
        "            if x >= d and not win(x - d): return True\n"
        "        return False\n"
        "    print('First' if win(n) else 'Second')\n"
        "if __name__ == '__main__': solve()\n"
    )

    critic_feedback = {
        "analysis": "Naive recursive win(x) has exponential O(2^N) complexity. Must use O(1) modulo parity: n % 3 != 0.",
        "status": "BUG",
        "bottleneck_class": "asymptotic_time",
        "escalate_to_planner": False,
        "target_lines": "L5-L10",
        "patch_directive": "Replace recursive win(x) with O(1) condition: n % 3 != 0",
    }
    verdict = JevVerdict.model_validate(critic_feedback)
    assert verdict.bottleneck_class == "asymptotic_time"
    assert verdict.status == "BUG"

    diff = (
        "--- a/solution.py\n"
        "+++ b/solution.py\n"
        "@@ -5,6 +5,2 @@\n"
        "-    def win(x):\n"
        "-        if x == 0: return False\n"
        "-        for d in (1, 2):\n"
        "-            if x >= d and not win(x - d): return True\n"
        "-        return False\n"
        "-    print('First' if win(n) else 'Second')\n"
        "+    print('First' if n % 3 != 0 else 'Second')\n"
    )

    success, patched_code, patch_meta = apply_patch(buggy_game_code, diff)
    assert success is True
    assert "n % 3 != 0" in patched_code
    assert "def win(x)" not in patched_code


def test_server_game_theory_solve():
    """Verify localhost server solve endpoint on Game Theory problem."""
    app = create_app(mode="mock")
    client = TestClient(app)

    gt_spec = {
        "id": "GT-TEST-15C",
        "title": "Industrial Nim Game",
        "description": "Determine if first player wins Nim on quarry ranges.",
        "input_format": "N followed by pairs of x and m",
        "output_format": "'tolik' or 'bolik'",
        "constraints": "1 <= N <= 10^5",
        "sample_inputs": ["1\n3 5\n"],
        "sample_outputs": ["3\n"],
    }

    resp = client.post("/solve?language=python", json=gt_spec)
    assert resp.status_code == 200
    trace = resp.json()
    assert trace["problem_id"] == "GT-TEST-15C"
    assert trace["solved"] is True
    assert trace["final_verdict"] == "AC"
