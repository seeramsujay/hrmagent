# HRM Agent API & CLI Reference Manual

This document details the command-line interface (CLI) and REST API endpoints for the **HRM Agent** system.

---

## 1. CLI Reference (`hrm`)

The CLI is powered by Typer and provides commands for data mining, model training, server deployment, and benchmarking.

### Global Options
- `--help`: Show usage instructions.

### Commands

#### `hrm serve`
Launches the localhost OpenAI-compatible and HRM reasoning server.

```bash
hrm serve [OPTIONS]
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--host` | `str` | `127.0.0.1` | Host address to bind. |
| `--port` | `int` | `8000` | Port to bind. |
| `--mode` | `str` | `mock` | Execution mode: `mock` (unit testing) or `local` (real Transformers weights). |
| `--device` | `str` | `auto` | Compute device (`cuda`, `cpu`, `mps`, or `auto`). |
| `--role1-path` | `str` | `outputs/merged/role1_architect_critic` | Path to merged Role 1 model/adapter. |
| `--role2-path` | `str` | `outputs/merged/role2_coder_patcher` | Path to merged Role 2 model/adapter. |
| `--api-key` | `str` | `None` | Optional Bearer API key required for incoming requests. |
| `--reload` | `bool` | `False` | Enable auto-reload on code change (preserves auth environment). |

#### `hrm 4m-solve`
Executes closed-loop problem solving using the complete 4-Tier Hierarchical Reasoning Architecture.

```bash
hrm 4m-solve [OPTIONS]
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--problem-file` | `str` | `examples/codechef_problem_example.json` | Path to input ProblemSpec JSON file. |
| `--language` | `str` | `python` | Target language (`python`, `cpp`, `typescript`, `kotlin`). |

#### `hrm terminalbench`
Executes the standardized 5-domain TerminalBench agent evaluation suite.

```bash
hrm terminalbench [OPTIONS]
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--output-report` | `str` | `outputs/terminal_bench_report.json` | Path to write benchmark report JSON. |

#### `hrm solve-maze`
Solves a 30x30 maze with Sapient's official HRM checkpoint.

```bash
hrm solve-maze MAZE_FILE [--device auto]
```

---

## 2. REST API Reference

The server exposes standard OpenAPI 3.1 documentation at `/docs` (Swagger UI) and `/openapi.json`.

### Health & Discovery
- `GET /`: Service metadata, version, and mode.
- `GET /health`: Server health check, active device, and resident models.
- `GET /openapi.json`: Official OpenAPI specification.

### HRM Reasoning Endpoints

#### `POST /v1/4m/solve`
Solves a problem spec using the 4-tier pipeline (HRM $\to$ Supervisor $\to$ Coder $\to$ Verifier).

- **Headers**: `Authorization: Bearer <API_KEY>` (if configured)
- **Request Body**:
  ```json
  {
    "id": "CC-FLOW017",
    "title": "Second Largest",
    "description": "Three numbers A, B, and C are given. Find second largest.",
    "constraints": "1 <= T <= 100, 1 <= A, B, C <= 10^6",
    "sample_inputs": ["2\n10 20 30\n500 100 250\n"],
    "sample_outputs": ["20\n250\n"]
  }
  ```
- **Response**:
  ```json
  {
    "problem_id": "CC-FLOW017",
    "solved": true,
    "final_verdict": "AC",
    "engine": "mock",
    "synthesized_code": "import sys\n...",
    "repair_iterations": 0
  }
  ```

#### `POST /solve`
Legacy endpoint executing the dual-pair state machine.

#### `POST /evaluate`
Executes candidate code in the sandboxed runner with `os.wait4` kernel telemetry.

- **Request Body**:
  ```json
  {
    "code": "print(int(input()) * 2)",
    "language": "python",
    "input_data": "21\n",
    "expected_output": "42\n",
    "time_limit_ms": 2000,
    "memory_limit_mb": 256
  }
  ```
- **Response**:
  ```json
  {
    "verdict": "AC",
    "exit_code": 0,
    "cpu_time_ms": 12.4,
    "peak_rss_kb": 9740,
    "stdout": "42\n",
    "stderr": ""
  }
  ```

### OpenAI Compatible Endpoints
- `GET /v1/models`: Lists active models (`role1_architect_critic`, `role2_coder_patcher`, `hrm-4m-reasoner`).
- `POST /v1/chat/completions`: Standard OpenAI chat completion interface. Compatible with Aider, SWE-bench, OpenHands, and lm-evaluation-harness.
- `POST /v1/completions`: Standard OpenAI text completion interface.
