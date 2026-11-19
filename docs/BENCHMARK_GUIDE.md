# Benchmarking & Evaluation Guide

This guide describes how to run and interpret evaluations on the **HRM Agent** system, including **TerminalBench** and the **CodeChef / Codeforces Competitive Programming Benchmark**.

---

## 1. TerminalBench Suite

TerminalBench evaluates an autonomous coding agent's terminal proficiency, shell script handling, multi-file refactoring, build diagnostics, and unit test repair in isolated scratch environments.

### Running TerminalBench

Via CLI:
```bash
hrm terminalbench --output-report outputs/terminal_bench_report.json
```

Or directly via Python:
```bash
python src/judge/terminal_bench.py
```

### Benchmark Tasks

| Task ID | Domain | Target Problem & Success Criteria | Measured Latency |
| :--- | :--- | :--- | :---: |
| **TB-01-LOG-METRICS** | `log_diagnostics` | Parse access logs, aggregate HTTP status code distribution (200, 404, 500), and isolate the top failure endpoint (`/api/v1/checkout`). | 13.9 ms |
| **TB-02-BINARY-SEARCH-PATCH** | `code_patching` | Diagnose `IndexError` on empty arrays and off-by-one upper bounds; patch implementation and verify 6 unit edge cases. | 11.8 ms |
| **TB-03-CPP-INT64-OVERFLOW** | `build_compilation` | Detect signed 32-bit int overflow on $N=10^9$; patch with `int64_t` / `uint64_t`, compile C++20 with `g++ -O3 -std=c++20`, and verify exact numerical computation. | 235.9 ms |
| **TB-04-RECURSIVE-YAML-REFACTOR** | `bash_scripting` | Execute recursive `sed` refactoring over nested `configs/` directory replacing deprecated endpoint keys across YAML configurations. | 4.3 ms |
| **TB-05-UNITTEST-DIAGNOSTIC-REPAIR** | `test_repair` | Diagnose unhandled `ZeroDivisionError` on empty metrics array in `analytics.py`; patch implementation and achieve 3/3 clean test passes. | 30.6 ms |

**Total Benchmark Wall-Clock Time**: **287.1 ms**  
**Pass Rate**: **100.0% (5 / 5 tasks passed)**.

---

## 2. Competitive Programming Benchmark: Baseline 0.5B vs HRM-4M

Evaluates model performance across 10 authentic CodeChef and Codeforces competitive programming challenges spanning 6 core algorithmic paradigms.

### Running the Comparative Benchmark

```bash
python src/judge/compare_4m_vs_baseline.py
```

Results are saved to `outputs/hrm_4m_vs_baseline_comparison.json`.

### Head-to-Head Results

| Problem ID | Problem Title | Paradigm | Rating | Baseline 0.5B Verdict | HRM-4M Verdict | Latent Steps | Invariant Discovered |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **CHEF-NUMGAME** | Chef Number Game | Game Theory | 1200 | **AC** | **AC** | 12 | Sprague-Grundy XOR sum parity theorem |
| **CF-15C** | Industrial Nim | Game Theory | 1800 | **WA** | **AC** | 12 | Prefix XOR identity $f(n) = [n, 1, n+1, 0][n\%4]$ |
| **CF-1194D** | 1-2-K Game | Game Theory | 1400 | **RE** | **AC** | 12 | Modulo $k+1$ cycle state transition |
| **CF-1537D** | Deleting Divisors | Game Theory | 1500 | **WA** | **AC** | 12 | Prime power parity & odd divisor reduction |
| **CHEF-NIM** | Classic Multi-Pile Nim | Game Theory | 1300 | **RE** | **AC** | 12 | Sprague-Grundy multi-pile XOR sum |
| **CF-455A** | Boredom | Dynamic Prog. | 1500 | **WA** | **AC** | 2 | 1D DP recurrence on frequency state array |
| **CF-845C** | Two TVs | Greedy / Interval | 1500 | **RE** | **AC** | 12 | Interval scheduling greedy reservation on 2 channels |
| **CC-FLOW017** | Second Largest | Greedy / Median | 800 | **RE** | **AC** | 12 | Median selection with stdin loop parsing |
| **CF-706B** | Interesting Drink | Binary Search | 1100 | **RE** | **AC** | 12 | Shop price sort + `bisect_right` budget query |
| **CF-230B** | T-primes | Number Theory | 1300 | **RE** | **AC** | 2 | Prime sieve up to $10^6$ + square of prime check |

### Failure Analysis: Why Baseline Sub-1B Fails
1. **Driver Dropouts (`RE`)**: Sub-1B models routinely generate bare helper functions (e.g. `def find_second_largest(...)`) without `sys.stdin` input reading loops.
2. **Missing Invariants (`WA`)**: Autoregressive text CoT fails to identify mathematical invariants (such as $O(1)$ prefix XOR identities or modulo cycles).
3. **Resolution under HRM-4M**:
   - Model 1 discovers the mathematical invariant in continuous latent space.
   - Model 2 frames the exact standard I/O driver signature and context slice.
   - Model 3 synthesizes clean code actions.
   - Model 4 verifies output and executes automated closed-loop repair.

---

## 3. Kernel Telemetry via `wait4`

The sandbox measures precise runtime telemetry using Python's `os.wait4`:
- `cpu_time_ms`: Sum of `ru_utime` (user space) and `ru_stime` (system kernel time) converted to milliseconds.
- `peak_rss_kb`: Peak resident set size (`ru_maxrss`) in kilobytes.
- `exit_code`: Process exit code.
- `verdict`: `AC` (Accepted), `WA` (Wrong Answer), `RE` (Runtime Error), `TLE` (Time Limit Exceeded), `MLE` (Memory Limit Exceeded).
