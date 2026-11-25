"""Tests for hrmagent Agent Client Protocol (ACP) server and client."""

import asyncio
import json
from pathlib import Path

from hrm_agent.acp_client import AcpLLMEngine, resolve_acp_command
from hrm_agent.acp_server import HrmAgentAcpServer, _extract_prompt_text


def test_hrm_acp_server_modes_and_commands(tmp_path):
    async def _run():
        out_file = tmp_path / "acp_out.jsonl"
        server = HrmAgentAcpServer(
            default_mode="4m",
            device="cpu",
            out_stream=open(out_file, "w", encoding="utf-8"),
        )
        try:
            init_res = await server.handle_message(
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": 1}}
            )
            assert init_res is not None
            assert init_res["agentInfo"]["name"] == "hrmagent"

            sess_res = await server.handle_message(
                {"jsonrpc": "2.0", "id": 2, "method": "session/new", "params": {"cwd": str(tmp_path), "mcpServers": []}}
            )
            assert sess_res is not None
            sid = sess_res["sessionId"]
            assert sess_res["modes"]["currentModeId"] == "4m"

            prob_json = json.dumps(
                {
                    "id": "TEST-CC-FLOW017",
                    "title": "Second Largest",
                    "description": "Find the second largest among A, B, C.",
                    "sample_inputs": ["2\n10 20 30\n500 100 250\n"],
                    "sample_outputs": ["20\n250\n"],
                }
            )
            p_res = await server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "session/prompt",
                    "params": {"sessionId": sid, "prompt": [{"type": "text", "text": prob_json}]},
                }
            )
            assert p_res == {"stopReason": "end_turn"}

            await server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "session/set_mode",
                    "params": {"sessionId": sid, "modeId": "code-action"},
                }
            )
            ca_res = await server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "session/prompt",
                    "params": {
                        "sessionId": sid,
                        "prompt": "```python\nwrite_file('out.txt', 'ok')\nprint(read_file('out.txt'))\n```",
                    },
                }
            )
            assert ca_res == {"stopReason": "end_turn"}
            assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "ok"

            inv_res = await server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 6,
                    "method": "session/prompt",
                    "params": {"sessionId": sid, "prompt": "/help"},
                }
            )
            assert inv_res == {"stopReason": "end_turn"}
        finally:
            server._out.close()

        msgs = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines() if line]
        chunks = [
            m["params"]["update"]["content"]["text"]
            for m in msgs
            if m.get("method") == "session/update"
            and m["params"]["update"].get("sessionUpdate") == "agent_message_chunk"
        ]
        joined = "\n".join(chunks)
        assert "Verdict:" in joined
        assert "/maze" in joined

    asyncio.run(_run())


def test_acp_llm_engine_end_to_end_stdio(tmp_path):
    hrm_bin = Path(__file__).resolve().parents[1] / "bin" / "hrmagent"
    engine = AcpLLMEngine(backend=f"{hrm_bin} acp --device cpu", cwd=str(tmp_path), timeout_sec=30.0)
    text, telem = engine.generate("/help")
    assert "/maze" in text
    assert telem.total_latency_ms > 0


def test_hrm_resolve_acp_command():
    agy_cmd = resolve_acp_command("agy")
    assert "acp" in agy_cmd
    claude_cmd = resolve_acp_command("claude")
    assert any("claude-code-acp" in part for part in claude_cmd)


def test_maze_prompt_keeps_trailing_free_cells():
    assert _extract_prompt_text("/maze\n# S \n") == "/maze\n# S "
    assert _extract_prompt_text([{"type": "text", "text": "/maze\n#G "}]) == "/maze\n#G "
    assert _extract_prompt_text("/MAZE\n#G ") == "/MAZE\n#G "
    assert _extract_prompt_text("  hello  ") == "hello"
