# AGENTS.md

Autonomous agent guide for interacting with `hrmagent` (HRM-4M).

---

## Quick Reference

- **CLI Binary (uses `uv`):** `./bin/hrmagent` (or `./bin/hrm`)
- **ACP Server (stdio JSON-RPC 2.0):** `hrmagent acp` (modes: `4m`, `dual-pair`, `code-action`; optional `--acp-backend agy|claude`)
- **MCP Server (stdio):** `hrmagent mcp`
- **OpenAI Model API:** `http://127.0.0.1:8000/v1` (Models: `hrm-dual-pair`, `hrm-architect-critic`, `hrm-coder-patcher`)
- **AGY Skill:** `.agents/skills/hrmagent/SKILL.md` (and `~/.gemini/config/skills/hrmagent/SKILL.md`)
- **Claude Skill:** `.claude/skills/hrmagent/SKILL.md` (and `~/.claude/skills/hrmagent/SKILL.md`)
- **MCP Configs:** `.agents/mcp_config.json` (AGY) and `.mcp.json` (Claude Code)

---

## Recommended Workflow for Agents

1. **Invariant Discovery (`hrm_discover_invariant`):** Extract mathematical/algorithmic paradigm and latent invariant before writing code.
2. **Closed-Loop Synthesis (`hrm_solve_4m` / `hrm_solve_loop`):** Synthesize solution code with System-2 blueprints and closed-loop sample verification.
3. **Sandboxed Verification (`hrm_verify_code` / `hrm_run_code_action`):** Validate patches against test cases with CPU/RSS `os.wait4` telemetry.
4. **ACP Delegation:** Use `hrmagent acp` directly from Zed / `remote-zed` / ACP clients, or pass `--acp-backend agy|claude` so `HRMAgentLoop` / `FourModelAgent` call external ACP agents.
