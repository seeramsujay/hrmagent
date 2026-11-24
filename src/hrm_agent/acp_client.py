"""ACP Client for hrmagent.

Provides `AcpLLMEngine`, an `LLMEngine` implementation backed by any Agent Client
Protocol (ACP) stdio agent (`agy acp`, `claude-code-acp`, or a custom command),
allowing HRMAgentLoop and FourModelAgent to delegate reasoning/coding turns over ACP.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import shutil
import time
from pathlib import Path
from typing import Any

from hrm_agent.llm_engine import LLMEngine
from hrm_agent.schemas import LLMTelemetry


def resolve_acp_command(backend: str) -> list[str]:
    """Resolve a named ACP backend ('agy', 'claude') or custom command into argv."""
    key = backend.strip().lower()
    if key in ("agy", "antigravity"):
        exe = shutil.which("agy") or "/snap/bin/agy"
        return [exe, "acp", "--dangerously-skip-permissions"]
    if key in ("claude", "claude-code", "claude-code-acp"):
        cc_acp = shutil.which("claude-code-acp")
        if cc_acp:
            return [cc_acp]
        npx = shutil.which("npx") or "npx"
        return [npx, "-y", "@zed-industries/claude-code-acp"]
    return shlex.split(backend)


def _extract_chunk_text(update: dict[str, Any]) -> str:
    upd_type = update.get("sessionUpdate") or update.get("sessionUpdateType") or update.get("type")
    if upd_type not in ("agent_message_chunk", "agent_message", None):
        return ""
    content = update.get("content")
    if isinstance(content, dict):
        return str(content.get("text", ""))
    if isinstance(content, list):
        return "".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
    return ""


async def _run_acp_prompt(
    command: list[str],
    prompt: str,
    cwd: str,
    timeout_sec: float,
) -> str:
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    assert proc.stdin is not None and proc.stdout is not None

    next_id = 1
    collected: list[str] = []

    async def send(method: str, params: dict[str, Any]) -> int:
        nonlocal next_id
        rid = next_id
        next_id += 1
        line = json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}) + "\n"
        proc.stdin.write(line.encode("utf-8"))  # type: ignore[union-attr]
        await proc.stdin.drain()  # type: ignore[union-attr]
        return rid

    async def wait_resp(target_id: int) -> dict[str, Any]:
        while True:
            raw = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout_sec)  # type: ignore[union-attr]
            if not raw:
                stderr_tail = ""
                if proc.stderr:
                    try:
                        stderr_tail = (await asyncio.wait_for(proc.stderr.read(4096), timeout=0.5)).decode(
                            "utf-8", errors="replace"
                        )
                    except Exception:
                        pass
                raise RuntimeError(f"ACP agent closed before replying to id={target_id}: {stderr_tail}")
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            if msg.get("method") == "session/update":
                upd = (msg.get("params") or {}).get("update") or {}
                chunk = _extract_chunk_text(upd)
                if chunk:
                    collected.append(chunk)
                continue

            if msg.get("id") == target_id:
                if "error" in msg:
                    raise RuntimeError(f"ACP error: {msg['error']}")
                return msg.get("result") or {}

    try:
        init_id = await send(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {"fs": {"readTextFile": False, "writeTextFile": False}, "terminal": False},
                "clientInfo": {"name": "hrmagent-acp-client", "version": "0.1.0"},
            },
        )
        await wait_resp(init_id)

        sess_id = await send("session/new", {"cwd": cwd, "mcpServers": []})
        sess_res = await wait_resp(sess_id)
        session_id = sess_res["sessionId"]

        prompt_id = await send(
            "session/prompt",
            {"sessionId": session_id, "prompt": [{"type": "text", "text": prompt}]},
        )
        await wait_resp(prompt_id)
        return "".join(collected).strip()
    finally:
        try:
            if proc.stdin:
                proc.stdin.close()
            if proc.returncode is None:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=2.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


class AcpLLMEngine(LLMEngine):
    """LLMEngine adapter that delegates generation to an external ACP agent."""

    def __init__(
        self,
        backend: str = "agy",
        cwd: str | None = None,
        timeout_sec: float = 90.0,
    ):
        self.backend = backend
        self.command = resolve_acp_command(backend)
        self.cwd = cwd or str(Path.cwd())
        self.timeout_sec = timeout_sec

    def generate(
        self,
        prompt: str,
        grammar_text: str | None = None,
        grammar_path: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop: list[str] | None = None,
    ) -> tuple[str, LLMTelemetry]:
        full_prompt = prompt
        if grammar_text:
            full_prompt += f"\n\n[Strict Output Grammar]\n{grammar_text}"

        t0 = time.perf_counter()
        text = asyncio.run(_run_acp_prompt(self.command, full_prompt, self.cwd, self.timeout_sec))
        total_ms = (time.perf_counter() - t0) * 1000.0

        prompt_tokens = max(1, len(full_prompt) // 4)
        completion_tokens = max(1, len(text) // 4)
        tok_s = (completion_tokens / (total_ms / 1000.0)) if total_ms > 0 else 0.0

        telemetry = LLMTelemetry(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prefill_ms=round(total_ms * 0.15, 2),
            decode_ms=round(total_ms * 0.85, 2),
            tokens_per_second=round(tok_s, 2),
            total_latency_ms=round(total_ms, 2),
        )
        return text, telemetry
