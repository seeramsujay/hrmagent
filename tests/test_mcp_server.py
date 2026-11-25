"""Tests for hrmagent Model Context Protocol (MCP) Server."""

import asyncio
import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from hrm_agent.mcp_server import HrmAgentMcpApp


def test_hrm_mcp_status_and_maze_errors():
    app = HrmAgentMcpApp(mode="mock", device="cpu")
    assert app.mcp.name == "hrmagent"

    st_raw = asyncio.run(app.mcp.call_tool("hrm_get_status", {}))
    assert not st_raw.is_error
    st = json.loads(st_raw.content[0].text)
    assert st["service"] == "hrmagent"

    # A bad grid is a tool error carrying the reason (clients receive isError: true).
    with pytest.raises(ToolError, match="30 lines"):
        asyncio.run(app.mcp.call_tool("hrm_solve_maze", {"maze": "too small"}))


def test_hrm_mcp_solve_4m_and_verify():
    app = HrmAgentMcpApp(mode="mock", device="cpu")

    res_raw = asyncio.run(
        app.mcp.call_tool(
            "hrm_solve_4m",
            {
                "problem_id": "TEST-CC-FLOW017",
                "title": "Second Largest",
                "description": "Three numbers A, B and C are given. Find the second largest among them.",
                "constraints": "1 <= T <= 100",
                "sample_inputs": ["2\n10 20 30\n500 100 250\n"],
                "sample_outputs": ["20\n250\n"],
                "language": "python",
            },
        )
    )
    assert not res_raw.is_error
    res = json.loads(res_raw.content[0].text)
    assert res["engine"] == "mock"
    assert res["solved"] is True
    assert res["final_verdict"] == "AC"

    verify_raw = asyncio.run(
        app.mcp.call_tool(
            "hrm_verify_code",
            {
                "code": "import sys\nprint(int(sys.stdin.read().strip()) * 2)\n",
                "language": "python",
                "sample_inputs": ["21\n"],
                "sample_outputs": ["42\n"],
            },
        )
    )
    assert not verify_raw.is_error
    vdata = json.loads(verify_raw.content[0].text)
    assert vdata["all_passed"] is True


def test_hrm_mcp_code_action_and_terminalbench(tmp_path):
    app = HrmAgentMcpApp(mode="mock", device="cpu")

    ca_raw = asyncio.run(
        app.mcp.call_tool(
            "hrm_run_code_action",
            {
                "python_code": "write_file('hello.txt', 'hrm-mcp')\nprint(read_file('hello.txt'))",
                "workspace_dir": str(tmp_path),
            },
        )
    )
    assert not ca_raw.is_error
    ca = json.loads(ca_raw.content[0].text)
    assert ca["success"] is True
    assert "hrm-mcp" in ca["output"]

    tb_raw = asyncio.run(app.mcp.call_tool("hrm_run_terminalbench", {}))
    assert not tb_raw.is_error
    tb = json.loads(tb_raw.content[0].text)
    assert tb["total_tasks"] == 5
    assert tb["passed_tasks"] >= 4
