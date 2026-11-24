# CLAUDE.md

Guidelines and reference for Claude (Claude Code, Claude Desktop, Cursor, and Anthropic API agents) working with or developing `hrmagent`.

---

## 1. Project Overview

`hrmagent` (HRM-4M) combines a Sapient-inspired dual-timescale latent recurrent reasoner, a System-2 supervisor planner, a Code-Action worker, and a sandboxed Jev verifier (`os.wait4` CPU & MaxRSS telemetry).

### Integration Modes:
1. **ACP Server (`hrmagent acp`):** Stdio JSON-RPC 2.0 server (`4m`, `dual-pair`, `code-action` modes).
2. **MCP Server (`hrmagent mcp`):** Exposes `hrm_discover_invariant`, `hrm_solve_4m`, `hrm_solve_loop`, `hrm_run_code_action`, `hrm_verify_code`, `hrm_run_terminalbench`, `hrm_get_status`.
3. **ACP Client (`--acp-backend claude|agy`):** Backs HRM planner/coder/critic nodes with Claude or Antigravity over ACP.
4. **OpenAI-Compatible HTTP Server (`hrmagent serve`):** Exposes `/v1/chat/completions`, `/v1/models`, `/solve`, `/evaluate`, `/v1/4m/solve`.

---

## 2. Essential Commands (via `uv`)

```bash
# Sync / create Python 3.12 environment
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e .

# Run tests
uv run --no-sync pytest tests/ -v

# Run ACP or MCP server
./bin/hrmagent acp
./bin/hrmagent mcp
```

---

## 3. Configuring as an MCP / ACP Server

### For Claude Desktop (`~/.config/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "hrmagent": {
      "command": "/path/to/hrmagent/bin/hrmagent",
      "args": ["mcp"]
    }
  }
}
```

### For Claude Code CLI:
```bash
claude mcp add hrmagent -- /path/to/hrmagent/bin/hrmagent mcp
```
Project-scoped `.mcp.json` and `.claude/skills/hrmagent/SKILL.md` are preconfigured in this repository.

### For Zed / ACP Clients (`~/.config/zed/settings.json`):
```json
{
  "agent_servers": {
    "HRMAgent": {
      "type": "custom",
      "command": "/path/to/hrmagent/bin/hrmagent",
      "args": ["acp"]
    }
  }
}
```
Or run `./bin/hrmagent setup-all` on the host machine to configure AGY, Claude, and Zed automatically.
