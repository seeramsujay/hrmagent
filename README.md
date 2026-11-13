# HRM Code Agent: Production-Grade Local-First Hierarchical Reasoning Model

A local-first, closed-loop Hierarchical Reasoning Model (HRM) competitive programming agent and evaluation suite engineered for:
- **Low-footprint CPU inference**: `~2.5 GB` total RAM envelope on P-cores with `q8_0` KV caching (`<100 MiB` total KV RAM).
- **Single-GPU fine-tuning**: `10 GB` VRAM (`RTX 3080`) `bf16` LoRA training using `adamw_8bit` and Liger chunked cross-entropy.
- **Strict grammar constraints**: GBNF-enforced JSON blueprints, raw C++20 emission without markdown backticks, and unified diff patches.
- **Zero cloud API calls at runtime**: Runs entirely on local `llama.cpp` resident processes or `llama-server`.

---

## 1. System Architecture

The agent executes a three-node hierarchical state machine:

```
                      +-----------------------------------+
                      |   Node 1: System 2 Planner        |
                      |   (1.5B Q4_K_M GGUF / <250 tok)   |
                      +-----------------+-----------------+
                                        |  Blueprint JSON
                                        v
+------------------+  Pass Samples    +-------------------+  Fail Samples   +-------------------+
| Node 3: Jev Gate | <----------------| Node 2: Fast Coder| --------------> | Error-to-Diff Loop|
| (1.5B Q4_K_M)    |                  | (0.5B Q5_K_M GGUF)|                 | (max_inner_iters=3|
+--------+---------+                  +-------------------+                 +---------+---------+
         |                                      ^                                     |
         | Patch Directive (escalate=false)     +-------------------------------------+
         +--------------------------------------|   Apply Unified Diff Patch
         |                                      
         +--------------------------------------------------------------------> Node 1 (Re-plan)
           Asymptotic / Algorithmic Flaw (escalate_to_planner=true)
```

1. **Node 1 — System 2 Algorithmic Planner (`1.5B Q4_K_M`)**:
   - Emits a dense `<250`-token JSON blueprint (`analysis` first so reasoning precedes constrained slots, `paradigm`, `data_structures`, `time_complexity`, `space_complexity`, `use_int64`, `edge_cases`, `steps`).
   - Enforced strictly by `grammars/planner.gbnf`. Never writes full source code.
2. **Node 2 — System 1 Fast Coder & Unified-Diff Patcher (`0.5B Q5_K_M`)**:
   - Runs in an inner micro-loop (`max_inner_iters = 3`).
   - **Plan-to-Code Mode** (`grammars/cpp_raw.gbnf` / `grammars/py_raw.gbnf`): Translates blueprint directly into C++20 (`#include` prefix lock, `ios::sync_with_stdio(false); cin.tie(nullptr);`, `long long` aliases, `'\n'`) without markdown fences.
   - **Error-to-Unified-Diff Mode** (`grammars/unified_diff.gbnf`): On compiler `stderr`, runtime crash, sample mismatch, or Node 3 patch directive, emits unified diffs (`--- a/solution.cpp` / `+++ b/solution.cpp`) applied via strict offset matching with fuzzy hunk fallback.
3. **Node 3 — Jev Critic & Complexity Optimizer (`1.5B Q4_K_M`)**:
   - Post-sample verifier (`max_critic_iters = 2`).
   - Inspects `(Problem Constraints + Current Code + Sandbox CPU/Wall Time & Peak RSS Telemetry)`.
   - Emits GBNF JSON verdict (`grammars/critic.gbnf`): `analysis`, `status` (`PASS` | `OPTIMIZE` | `BUG`), `bottleneck_class`, `escalate_to_planner`, `target_lines`, `patch_directive`.

---

## 2. Directory Structure

```text
hrmagent/
├── pyproject.toml                     # Managed via `uv` (llama-cpp-python, datasets, duckdb, peft, trl, bitsandbytes, liger-kernel, pydantic, rich, typer, pytest)
├── README.md                          # Full guide and architecture specs
├── configs/
│   ├── models.yaml                    # Topologies (Option A / Option B), GGUF paths, q8_0 KV cache, CPU masks
│   └── training.yaml                  # 10GB RTX 3080 bf16 LoRA hyperparameters + curriculum bounds
├── grammars/
│   ├── planner.gbnf                   # JSON blueprint grammar (analysis first, paradigm enum, complexity regex)
│   ├── cpp_raw.gbnf                   # Forces `#include` start and blocks backticks (`[^\x60]+`)
│   ├── py_raw.gbnf                    # Blocks backticks for clean Python 3 emission
│   ├── unified_diff.gbnf              # Strict unified diff grammar (--- a/..., +++ b/..., @@ -l,s +l,s @@)
│   └── critic.gbnf                    # JSON verdict grammar (analysis first, PASS|OPTIMIZE|BUG, bottleneck_class)
├── src/hrm_agent/
│   ├── schemas.py                     # Pydantic v2 schemas: ProblemSpec, PlannerBlueprint, ExecutionTelemetry, JevVerdict, AgentTrace
│   ├── llm_engine.py                  # llama-cpp-python / llama-server client with q8_0 KV cache and GBNF caching
│   ├── patcher.py                     # Unified-diff applier (strict line-offset matching + fuzzy hunk fallback + syntax validation)
│   ├── hrm_loop.py                    # Core state machine: Macro-Step -> Micro-Loop -> Jev Gate -> Escalation
│   └── cli.py                         # Typer CLI: mine-data, train-lora, export-gguf, solve, eval-benchmark
├── src/judge/
│   ├── sandbox.py                     # Dual-stage runner: nsjail -> bwrap -> subprocess+resource.setrlimit with os.wait4 telemetry
│   ├── fuzz_oracle.py                 # Fuzz generator (boundary, N_max, 10^9 overflow, star/line graphs) + consensus oracle
│   └── benchmark_eval.py              # Stratified benchmark evaluator & 3-way ablation suite
├── src/training/
│   ├── dataset_miner.py               # HF + DuckDB curation: temporal cutoff, 8-gram decontamination, trajectory mining
│   ├── train_sft_lora.py              # bf16 LoRA (r=32, alpha=64, adamw_8bit, Liger chunked CE for <6.5 GB VRAM) + DPO
│   └── export_gguf.py                 # PEFT merge, calibration_cp.txt builder, llama-imatrix + llama-quantize, parity gate
└── tests/
    └── test_hrm_pipeline.py           # Pytest suite with mock LLM verifying grammars, patching, sandbox, and ablation
```

---

## 3. Quickstart & Installation

Manage dependencies using `uv`:

```bash
# Clone and enter directory
cd hrmagent

# Create Python 3.12 environment
uv venv .venv
source .venv/bin/activate

# Install dependencies in editable mode
uv pip install -e .
```

---

## 4. CLI Workflow

The `hrm` CLI exposes all lifecycle stages:

### Step 1: Mine & Curate Datasets
```bash
# Mines Codeforces and CodeChef trajectories and outputs balanced training splits
hrm mine-data --output-dir data/curated --min-rating 800 --max-rating 1900
```

### Step 2: Fine-Tune bf16 LoRA on 10 GB VRAM
```bash
# Trains 1.5B/0.5B Qwen model with adamw_8bit and Liger chunked cross-entropy (<6.5 GB VRAM)
hrm train-lora --config-path configs/training.yaml --mode sft --output-dir outputs/checkpoints

# (Optional) Train DPO alignment with adapter-disabled reference policy
hrm train-lora --mode dpo --output-dir outputs/checkpoints_dpo
```

### Step 3: Export & Quantize with Importance Matrix
```bash
# Merges LoRA adapter, generates domain calibration_cp.txt, and quantizes via llama-imatrix
hrm export-gguf --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct --output-gguf models/gguf/planner_q4_k_m.gguf --quant-type Q4_K_M
```

### Step 4: Solve a Competitive Programming Problem
```bash
# Solve an individual problem specification with the full HRM loop
hrm solve --problem-file examples/problem_example.json --language cpp
```

### Step 5: Run Benchmark & 3-Way Ablation Evaluation
```bash
# Evaluates Pass@1 across rating buckets (<1200, 1200-1500, 1500-1800, 1800-2100, 2100+)
# Compares Node 2 Only vs Node 1+2 vs Full HRM
hrm eval-benchmark --report-path outputs/eval_report.json
```

---

## 5. Verification Suite

Run the full end-to-end test suite:

```bash
.venv/bin/pytest tests/test_hrm_pipeline.py -v
```
