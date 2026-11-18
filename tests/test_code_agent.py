"""Test suite for Code-Action Agent and SEARCH / REPLACE diff patcher."""

from hrm_agent.code_agent import CodeActionAgent
from hrm_agent.patcher import apply_patch, parse_search_replace_blocks


def test_search_replace_diff_patching():
    """Verify SEARCH / REPLACE diff blocks replace code cleanly without fragile line offsets."""
    original_code = "def compute(a, b):\n" "    result = a + b\n" "    print('debug: summing')\n" "    return result\n"

    diff = (
        "<<<<<<< SEARCH\n"
        "    result = a + b\n"
        "    print('debug: summing')\n"
        "=======\n"
        "    result = a * b\n"
        ">>>>>>> REPLACE\n"
    )

    blocks = parse_search_replace_blocks(diff)
    assert len(blocks) == 1
    assert "result = a + b" in blocks[0][0]
    assert "result = a * b" in blocks[0][1]

    success, patched, meta = apply_patch(original_code, diff, language="python")
    assert success is True
    assert "result = a * b" in patched
    assert "print('debug: summing')" not in patched
    assert "def compute(a, b):" in patched


def test_code_action_agent_thinking_budget():
    """Verify thinking tags extraction and token budget capping."""
    agent = CodeActionAgent(thinking_budget=10)

    raw_text = (
        "<think>\n"
        "Thinking about problem complexity and potential off-by-one errors in loop.\n"
        "Need to check if list is empty or single element.\n"
        "</think>\n"
        "```python\n"
        "write_file('solution.py', 'print(42)')\n"
        "```"
    )

    think, executable, budget_used = agent.extract_thinking(raw_text)
    assert "Thinking about problem complexity" in think
    assert "write_file" in executable
    assert "<think>" not in executable
    assert budget_used <= 10


def test_code_action_execution_in_workspace(tmp_path):
    """Verify CodeActionAgent executes python snippets with atomic tools."""
    agent = CodeActionAgent(workspace=tmp_path)

    # 1. Write file via code action
    action1 = "write_file('test.txt', 'hello from code-action')"
    res1 = agent.execute_code_action(action1)
    assert res1.success is True
    assert (tmp_path / "test.txt").read_text() == "hello from code-action"

    # 2. Apply SEARCH / REPLACE diff via code action
    action2 = (
        "diff = '''<<<<<<< SEARCH\n"
        "hello from code-action\n"
        "=======\n"
        "hello from patched code-action\n"
        ">>>>>>> REPLACE'''\n"
        "apply_diff('test.txt', diff)\n"
    )
    res2 = agent.execute_code_action(action2)
    assert res2.success is True
    assert (tmp_path / "test.txt").read_text() == "hello from patched code-action"

    # 3. Read file and run command
    action3 = (
        "content = read_file('test.txt')\n"
        "print('READ:', content)\n"
        "out = run_command('wc -w test.txt')\n"
        "print('WC:', out.split()[0])\n"
    )
    res3 = agent.execute_code_action(action3)
    assert res3.success is True
    assert "READ: hello from patched code-action" in res3.output
    assert "WC: 4" in res3.output
