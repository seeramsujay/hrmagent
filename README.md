<p align="center">
  <img src="assets/logo.svg" alt="hrmagent Logo" width="120" height="120" />
</p>

<h1 align="center">hrmagent</h1>

<p align="center">
  <em>Local coding agent (planner, coder, sandbox verifier) with Sapient HRM as a maze-solving tool</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Maintained-success?style=flat-square" alt="Status" />
  <img src="https://img.shields.io/badge/License-Protected-blue?style=flat-square" alt="License" />
</p>

---

# hrmagent

A local coding agent: a **supervisor** plans, a **coder** writes, and a **sandbox verifier** runs the samples
and asks for a patch. It also exposes Sapient's official [HRM](https://github.com/sapientinc/HRM) as a tool.

## What HRM does here, and what it does not

HRM maps a fixed token grid to a solved grid, with one checkpoint per task. It does not read text, discover
invariants, or write code, so it plays no part in code synthesis. `hrmagent` uses the official 30x30 maze
checkpoint through the sibling `PALAIS` repository:

```bash
hrm solve-maze maze.txt      # 30 lines of '#', ' ', 'S', 'G'
```

Also available as MCP tool `hrm_solve_maze` and ACP command `/maze`. Needs `../PALAIS` with
`third_party/HRM`, the maze checkpoint (`PALAIS/docs/reproduce.md`), torch, numpy, einops and pydantic.

## Retracted claims

Earlier versions reported 10/10 on CodeChef/Codeforces and described an HRM latent reasoner. The reasoner was
an untrained network with keyword matching, and the solutions were hand-written and selected by problem name.
Both are removed, and those numbers are void. The TerminalBench pass rate runs scripts bundled with the tasks,
so it tests the harness, not an agent. Command names containing `4m` are kept for compatibility.

Without a configured model the agent uses a **mock engine** with canned output.

## Installation & Quickstart

### Prerequisites
- Python 3.10+ (Python 3.12 recommended)
- `uv` (recommended) or standard `pip`
- NVIDIA GPU with CUDA 12+ (optional for GPU training/acceleration; CPU fallback supported)

### Setup
```bash
# Clone the repository
git clone https://github.com/seeramsujay/hrmagent.git
cd hrmagent

# Create virtual environment and install in editable mode
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

---

## Usage

```bash
hrm solve-maze maze.txt
hrm 4m-solve --problem-file examples/codechef_problem_example.json --acp-backend claude
hrm serve --host 127.0.0.1 --port 8000 --mode mock
hrm terminalbench
```

Training jobs are listed in `training/README.md` at the monorepo root.

---

## Documentation Links

- [Architecture & Latent Recurrence Specification](docs/ARCHITECTURE.md)
- [API & CLI Reference Manual](docs/API_REFERENCE.md)
- [Benchmarking & Evaluation Guide](docs/BENCHMARK_GUIDE.md)
- [Academic Research Preprint (HRM-4M)](docs/HRM_FOUR_MODEL_PAPER.md)
- [Comprehensive Hardware Evaluation Report](reports/benchmark_report.md)

---

## Testing & Linting

Run full verification:
```bash
pytest tests/ -v

# Run linters
black --check src tests
ruff format --check src tests
ruff check src tests
flake8 src tests
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
