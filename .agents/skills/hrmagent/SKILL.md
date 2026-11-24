---
name: hrmagent
description: >-
  Use this skill for hierarchical latent algorithmic reasoning, 4-model code synthesis
  (HRM Latent Reasoner -> Supervisor -> Code-Action Worker -> Jev Verifier), sandboxed
  execution with os.wait4 telemetry, TerminalBench diagnostics, or integrating hrmagent
  via ACP, MCP, CLI, or OpenAI-compatible HTTP endpoints.
---

# hrmagent (HRM-4M): Hierarchical Reasoning Coding Agent

`hrmagent` is a local-first autonomous coding and reasoning agent built for <= 6 GB VRAM GPUs and CPU fallback. It exposes:
1. **Agent Client Protocol (ACP):** Stdio JSON-RPC 2.0 server (`hrmagent acp`) supporting `4m`, `dual-pair`, and `code-action` modes, plus `AcpLLMEngine` client support for `agy` and `claude`.
2. **Model Context Protocol (MCP):** Exposes HRM latent invariant discovery, 4-model solving, 3-node loop solving, sandboxed code-actions, `os.wait4` verification, and TerminalBench over stdio/HTTP.
3. **Universal CLI (`hrmagent` / `hrm`):** Managed via `uv` (`uv run --project . ...`).
4. **OpenAI-Compatible Server:** `http://127.0.0.1:8000/v1` (`hrmagent serve --mode mock|dual-pair`).

---

## 1. Using hrmagent via MCP Tools

- `hrm_discover_invariant(problem_description, constraints="")`: Runs Sapient-style dual-timescale latent recurrence (`HRMLatentSolver`) to identify paradigm and mathematical invariant.
- `hrm_solve_4m(problem_id, description, title="", constraints="", sample_inputs=None, sample_outputs=None, language="python")`: Runs the 4-tier pipeline (`FourModelAgent`).
- `hrm_solve_loop(problem_id, description, title="", constraints="", sample_inputs=None, sample_outputs=None, language="python")`: Runs the 3-node closed-loop (`HRMAgentLoop`) with GBNF grammars and unified-diff repair.
- `hrm_run_code_action(python_code, workspace_dir=".")`: Executes workspace-confined Python code-actions (`read_file`, `write_file`, `run_command`, `apply_diff`).
- `hrm_verify_code(code, language="python", sample_inputs=None, sample_outputs=None)`: Runs code in `SandboxRunner` and returns per-case verdicts (`AC`, `WA`, `TLE`, `MLE`, `RE`, `CE`) and `os.wait4` telemetry.
- `hrm_run_terminalbench()`: Executes the 5-task TerminalBench suite.
- `hrm_get_status()`: Reports active device, CUDA status, sandbox engine, and grammar directory.

---

## 2. Using hrmagent via ACP and CLI

Binary path: `./bin/hrmagent` (alias `./bin/hrm`).

```bash
# Start ACP stdio JSON-RPC 2.0 server (default mode: 4m)
./bin/hrmagent acp
./bin/hrmagent acp --mode dual-pair --acp-backend agy
./bin/hrmagent acp --mode code-action

# Start MCP stdio server
./bin/hrmagent mcp

# Run 4-Model solver using Claude or AGY over ACP
./bin/hrmagent 4m-solve --problem-file examples/codechef_problem_example.json --acp-backend claude

# Run TerminalBench
./bin/hrmagent terminalbench

# Launch OpenAI-compatible localhost server
./bin/hrmagent serve --host 127.0.0.1 --port 8000 --mode mock
```

ACP slash commands in `session/prompt`:
- `/mode <4m|dual-pair|code-action>`
- `/invariant <problem>`
- `/terminalbench`
- `/status`
- `/help`
