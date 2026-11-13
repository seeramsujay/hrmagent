# **Engineering a CPU-Executable Sub-2B Hierarchical Reasoning Coding Agent for CodeChef and Codeforces**

Building a CPU-executable competitive programming agent from quantized sub-2B models requires decomposing monolithic code generation into three temporally separated cognitive roles inspired by dual-process theory and Sapient's Hierarchical Reasoning Model (HRM): a slow System 2 Algorithmic Planner (1.5B) emitting compact structured blueprints under 250 tokens, a fast System 1 Coder and Unified-Diff Patcher (0.5B) executing deterministic C++20 or Python code generation without conversational prose, and a diagnostic Jev-style Critic (1.5B) driven by execution telemetry from a local sandboxed judge1. Deployed via llama.cpp with Q4\_K\_M or Q5\_K\_M weights, q8\_0 KV caching, GBNF grammar constraints, and P-core thread pinning, this three-node pipeline targets an approximately 2.5 GB CPU RAM envelope while isolating untrusted binaries inside an nsjail or bubblewrap and seccomp-bpf harness5.  
Benchmark evidence shows why closed-loop execution feedback and structural specialization are mandatory at this parameter scale. In DeepSeek's January 2025 evaluation (as tabulated in secondary benchmark summaries and official DeepSeek model documentation), DeepSeek-R1-Distill-Qwen-1.5B achieved a 954 Codeforces rating—exceeding GPT-4o-0513 (759) and Claude-3.5-Sonnet-1022 (717)—yet reached only 16.9% Pass@1 on LiveCodeBench versus 38.9% for Claude-3.5-Sonnet-102210. Moreover, no independent benchmark in the retrieved 2024–2026 literature evaluates a three-node quantized sub-2B agent head-to-head against current Claude Haiku or Sonnet releases on CodeChef or Codeforces; narrowing that Pass@1 deficit depends on decoupling high-level planning from low-level syntax emission, replacing full-file rewrites with unified diffs, and scaling test-time compute over deterministic local verification2.

## **1\. Hierarchical Multi-Agent System 1 and System 2 Coding Architectures at Sub-2B Scale**

### **Research Trajectory in Dual-Process Reasoning and Sub-2B Agents**

Dual-process cognitive theory divides reasoning into fast, heuristic System 1 processing and slower, analytical System 2 deliberation—a separation surveyed across reasoning language models by Guo et al. (November 2025, *IEEE Transactions on Pattern Analysis and Machine Intelligence*) and formalized in Kahneman-inspired LLM taxonomies1. In June 2025 (revised September 2026), Wang et al. at Sapient introduced the *Hierarchical Reasoning Model* (HRM), pairing a slow high-level recurrent planner (![][image1], analogous to cortical theta rhythms at 4–8 Hz) with a fast low-level recurrent execution module (![][image2], analogous to gamma rhythms at 30–100 Hz)3. Using deep supervision, Post-Norm RMSNorm blocks, AdamW optimization, and Adaptive Computational Time (ACT) halting, HRM discovered complex algorithms from as few as 1,000 training examples and demonstrated strong inference-time scaling on long-horizon planning tasks such as Sudoku4. While HRM couples those timescales inside continuous latent recurrence, modular agent systems map the same temporal separation onto discrete token-emitting language models1.  
Recent 2026 studies demonstrate why decoupling planning from code generation is essential below 2B parameters. In rSIM (*ACL 2026*), Chen et al. showed that directly applying Group Relative Policy Optimization (GRPO) to Qwen2.5-0.5B causes rewards to climb briefly toward 0.3 before collapsing abruptly to zero, reinforcing Yue et al.'s (2025) observation that pure RL fails to induce absent reasoning strategies in capacity-limited base models2. Decoupling strategy generation into a separate planner continuously fine-tuned on CodeAlpaca-20k raised Qwen2.5-0.5B code generation accuracy by 17%, building on dual-agent pioneer-and-observer setups like CORY (Ma et al., 2024\)2. Complementary 2025–2026 approaches—including RECRL, which raised Qwen2.5-Coder-3B average Pass@1 across HumanEval, HumanEval+, MBPP, MBPP+, and LiveCodeBench from 60.07% to 66.50% (surpassing Qwen2.5-Coder-7B's 66.05%)19, SRI-Coder (*ACL 2026*), which replaced brittle Fill-in-the-Middle completion with corrective unified-diff editing15, and verification-guided cascades (V\_1, PRISM, and consensus-escalation pipelines)14—motivate the three-node sub-2B competitive coding architecture below (noting that while the "Jev-style" reviewer name reflects this system's diagnostic critic role, no published 2024–2026 paper evaluates a named "Jev" module directly):

> 1. **Node 1 (System 2 Algorithmic Planner, 1.5B)**: DeepSeek-R1-Distill-Qwen-1.5B or Qwen2.5-Coder-1.5B-Instruct distills problem constraints, algorithmic paradigms, asymptotic complexity targets, and overflow risks into a terse JSON blueprint (\<250 tokens).
> 2. **Node 2 (System 1 Fast Coder and Diff Patcher, 0.5B)**: Qwen2.5-Coder-0.5B-Instruct (0.49B total, 0.36B non-embedding parameters, 24 layers, 14 query and 2 key-value GQA heads, 32K context, Apache 2.022) converts blueprints into C++20 or Python code without chain-of-thought prose and applies localized unified-diff patches.
> 3. **Node 3 (Jev-Style Critic and Complexity Optimizer, 1.5B)**: Qwen2.5-Coder-1.5B-Instruct (1.54B total, 1.31B non-embedding parameters, 28 layers, 1,536 hidden size, 8,960 intermediate size, 128 head dimension, 12 query and 2 key-value GQA heads, Apache 2.022) evaluates sandbox compiler errors, sample diffs, and TLE/MLE telemetry to route localized bugs back to Node 2 or flawed algorithmic complexity back to Node 1\.

| Model | Total / Non-Emb Params | AIME 2024 Pass@1 | MATH-500 Pass@1 | HumanEval / MBPP | LiveCodeBench Pass@1 | Codeforces Elo | Vintage / License |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-Coder-0.5B-Instruct | 0.49B / 0.36B | —   | —   | 61.6 (57.3+) / 52.4 (43.7+) | —   | —   | Nov 2024 / Apache 2.022 |
| Qwen2.5-Coder-1.5B-Instruct | 1.54B / 1.31B | —   | —   | —   | —   | —   | Sep 2024 / Apache 2.022 |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.8B reported | 28.9% | 83.9% | —   | 16.9% | 954 | Jan 2025 / MIT12 |
| DeepSeek-R1-Distill-Qwen-7B | \~7.6B | 55.5% | 92.8% | —   | 37.6% | 1189 | Jan 2025 / MIT13 |
| GPT-4o-0513 | Closed | 9.3% | 74.6% | —   | 32.9% | 759 | May 2024 / Proprietary10 |
| Claude-3.5-Sonnet-1022 | Closed | 16.0% | 78.3% | —   | 38.9% | 717 | Oct 2024 / Proprietary11 |
| OpenAI-o1-1217 | Closed | 79.2% | 96.4% | —   | 63.4% | 2061 | Dec 2024 / Proprietary13 |

(Benchmark note: Values for DeepSeek-R1-Distill-Qwen-1.5B LiveCodeBench and Codeforces trace to DeepSeek's January 2025 release table as compiled in secondary summaries12, with matching GPT-4o-0513 and Claude-3.5-Sonnet-1022 entries confirmed in primary DeepSeek repositories10. Comparable Claude Haiku LiveCodeBench or Codeforces Elo scores were absent from the reviewed benchmark tables, where Claude 3 Haiku appears only as a peer comparison label28.)

### **Runtime Architecture: Shared-Backbone Multi-LoRA vs. Three Resident GGUF Models**

Deciding between hot-swapping LoRA adapters over a shared 1.5B backbone and keeping quantized GGUF processes resident in CPU RAM depends directly on llama.cpp attention-projection mechanics. In llama.cpp's llama-server (introduced via PR \#8332 and expanded in PR \#10994), multiple LoRA adapters load at startup with \--lora-scaled 0.0 \--lora-init-without-apply and switch dynamically via POST /lora-adapters5. However, as llama.cpp maintainer @ngxson detailed, every cached key and value vector is computed under the active effective weight matrix:  
![][image3]  
Changing adapter scale ![][image4] invalidates stored KV cache entries because cached projections reflect previous adapter weights, forcing a full KV-cache flush5. Furthermore, parallel server slots share batched embedding multiplications, preventing different slots from applying different LoRA scales simultaneously5. For alternative CPU runtimes, no benchmark or API documentation measuring ONNX Runtime multi-LoRA switching latency was found in the reviewed sources, whereas OpenVINO GenAI documents runtime LoRA adapter swapping alongside Linux llama.cpp acceleration across Intel CPU, GPU, and NPU targets31.  
Exact per-token KV cache sizing follows the standard Grouped-Query Attention (GQA) equation32:  
![][image5]  
For Qwen2.5-Coder-1.5B-Instruct (28 layers, 2 KV heads, 128 head dimension24), FP16 (2 bytes) consumes 2 × 28 × 2 × 128 × 2 \= 28,672 bytes/token (28 KiB/token), totaling 56 MiB at a 2,048-token context or 112 MiB at 4,096 tokens. Quantizing the KV cache to q8\_0 (\~1 byte/element plus small block-scale overhead) roughly halves this to \~28–30 MiB per 2,048-token 1.5B slot. Crucially, a 2026 engineering blog benchmark (particula.tech) notes that mainline llama.cpp builds on or after **2026-04-01** apply a normalized Hadamard rotation to Q, K, and V before caching; this bounds q8\_0 degradation, whereas q4\_0 KV caching suffers a sharp collapse from 81.6% to 8.3% similarity against f16 on a 12-prompt code, math, and reasoning suite6. For the 0.5B model (24 layers, 2 KV heads22; head\_dim unprinted in retrieved model cards), assuming head\_dim \<= 128 keeps combined three-node q8\_0 KV residency below \~100 MiB at 2,048 tokens.

| Runtime Topology | Weight Footprint (GGUF Verified & Est.) | KV Cache & Prefix Caching | Latency & Engineering Trade-Off |
| --- | --- | --- | --- |
| **Option A: Three Resident Quantized Processes** (R1-Distill-1.5B Q5\_K\_M \+ Coder-0.5B \+ Coder-1.5B) | 1.29 GB (Second State R1-1.5B Q5\_K\_M33) \+ 398 MB (Ollama Coder-0.5B34) \+ \~0.9–1.1 GB (est. Coder-1.5B Q4\_K\_M/Q5\_K\_M) \= **\~2.6–2.8 GB** (\~2.4–2.5 GB if both 1.5B use Q4\_K\_M) | Separate q8\_0 KV slots per node (\<100 MiB total at 2K); **preserves system-prompt and problem-statement prefix caching** across repair loops5. | Zero adapter-switch overhead and heterogeneous backbones (DeepSeek-R1-Distill Planner \+ Qwen2.5-Coder Critic), at highest RAM cost. |
| **Option B: Hybrid (Shared Qwen2.5-Coder-1.5B Multi-LoRA \+ Resident 0.5B Coder)** | \~0.9–1.1 GB (est. 1.5B base \+ Planner/Critic LoRAs) \+ 398 MB (0.5B34) \= **\~1.3–1.5 GB** (est.) | 0.5B Coder maintains unbroken KV prefix cache during diff loops; 1.5B flushes KV only when swapping Planner \<-\> Critic LoRAs5. | Strongest fit when CPU RAM is capped near 1.6–2.0 GB; requires training Planner on Qwen2.5-Coder-1.5B instead of R1-Distill-1.5B. |
| **Option C: Single-Backbone Multi-LoRA (1.5B \+ 3 LoRAs)** | **\~1.0–1.2 GB** (est.) | Every Planner \-\> Coder \-\> Critic transition triggers KV-cache invalidation and full prompt re-prefill5. | Minimum RAM footprint, but incurs repeated prefill latency and forfeits 0.5B decode throughput on System 1 loops. |

### **CPU Thread-Pinning on Hybrid x86\_64 Architectures**

Autoregressive CPU token generation is memory-bandwidth bound, synchronizing all ggml worker threads after every matrix-vector row reduction. On hybrid x86\_64 CPUs mixing Performance cores (P-cores), Efficiency cores (E-cores), and Simultaneous Multithreading (SMT/Hyperthreading), scheduling ggml workers across asymmetric cores stalls fast P-cores waiting for slower E-cores while SMT siblings contend for shared L1/L2 caches and vector units. Mainline llama.cpp prevents this bottleneck via \--cpu-mask and ggml threadpool affinity, explicitly designed to bind inference threads strictly to physical P-cores while excluding E-cores and Hyperthreaded siblings7.  
In an agentic competitive-programming pipeline (e.g., an 8 P-core \+ 8 E-core desktop CPU), pin the active llama-server process solely to the physical P-core bitmask (-t \<num\_physical\_p\_cores\> \--cpu-mask \<p\_core\_hex\_mask\>) while binding sandboxed g++ \-O2 compilation and local judge execution via taskset onto E-cores. Because the sequential Planner \-\> Coder \-\> Sandbox \-\> Critic workflow executes only one LLM node at a time, resident multi-process deployments avoid cross-model core contention while keeping compilation cache traffic isolated from P-core L2/L3 caches.

### **Inference-Time Scaling Laws and Unified-Diff Patching vs. Full-File Generation**

Test-time compute scaling—combining parallel Best-of-N sampling, execution-guided filtering, and sequential tree search over compiler and test feedback—enables sub-2B models to exceed greedy Pass@1 limits14. Following dual-mode inference frameworks14, parallel sampling (width) draws multiple candidate blueprints from Node 1 and multiple low-cost implementations from Node 2 at nonzero temperature (0.6), filtering candidates deterministically through sandbox execution against public and fuzzed tests rather than relying on ungrounded majority voting alone14. Asymmetric width allocation—generating few 1.5B plans and branching more 0.5B code candidates—concentrates CPU FLOPs in the smaller System 1 model.  
Sequential refinement (depth) operates over compiler stderr, runtime errors, and sample mismatch diffs14. When Node 3 diagnoses a localized error such as an integer overflow or off-by-one loop bound, Node 2 emits a **Unified-Diff Patch** rather than regenerating the entire source file. In SRI-Coder (*ACL 2026*), standard Fill-in-the-Middle completion scored 0.0 EM in noisy repair contexts, whereas unified-diff editing successfully isolated corrective changes15. On CPU, because decode latency scales linearly with generated tokens, emitting only changed diff hunks eliminates the token cost of regenerating fast-I/O headers, type aliases, and verified helper functions while preventing the 0.5B model from accidentally breaking unrelated lines. Control escalates back up the search tree to Node 1 (escalate\_to\_planner: true) only when Node 3 identifies a structural algorithm or asymptotic complexity flaw.

## **2\. CodeChef and Codeforces Dataset Curation, Difficulty Curriculum, and Trajectory Mining**

### **Verifiable Hugging Face Corpora and Exact Schemas**

All training, distillation, trajectory mining, and verification datasets can be assembled directly from Hugging Face Parquet and JSONL repositories without web scraping. Table 2 compiles the verified scales, schemas, licenses, and roles across the seven primary corpora.

| Dataset | Scale & Size | Verified Schema Columns & Subsets | Coverage & Vintage | Target Node & Role |
| --- | --- | --- | --- | --- |
| open-r1/codeforces37 | \~10k problems (train: \~9.55k, test: 468; verifiable: 8.76k; verifiable-prompts: \~16.1k); \~110 GB external generated\_tests/\*.parquet (cc-by-4.0) | id, aliases, contest\_id, contest\_name, contest\_type (ICPC/CF/IOI), contest\_start, contest\_start\_year, index, time\_limit, memory\_limit, title, description, input\_format, output\_format, interaction\_format, note, examples, editorial (\~60%), rating, tags, testset\_size, official\_tests, official\_tests\_complete, input\_mode (stdio/file), generated\_checker (checker.py), executable, generated\_tests; external tests have problem\_id, input, output, test\_case\_i (verifiable-prompts adds language, prompt) | Codeforces through early 2025; LaTeX OCR via Qwen2.5-VL-7B-Instruct | Held-out evaluation, temporal leakage anchor, oracle/checker verification |
| open-r1/codeforces-cots40 | 9.69 GB across 11 subsets (solutions: 47.8k, solutions\_decontaminated: 40.7k, solutions\_w\_editorials: 29.2k, solutions\_py: 9.56k, test\_input\_generator: 20.6k, checker\_interactor: 35.7k, etc.) (ODC-By 4.0) | messages column ready for SFT; DeepSeek-R1 C++ and Python traces; \_decontaminated subsets strip 8-gram overlap against AIME24/25, GPQA Diamond, MATH-500, LiveCodeBench. Unfiltered (\~84% Python public-test pass rate) | Codeforces (Mar 2025\) | Node 1 blueprint distillation (after execution gating) & fuzz-generator training |
| open-r1/codeforces-submissions38 | 12.7M human submissions (5.64 GB Parquet) across 9,906 problems (default, selected\_accepted: 9,183 probs, selected\_incorrect: 2,385 probs) (cc-by-4.0) | submission\_id, source, contestId, problem\_index, problem\_id, programmingLanguage, verdict (OK, TIME\_LIMIT\_EXCEEDED, WRONG\_ANSWER, MEMORY\_LIMIT\_EXCEEDED, RUNTIME\_ERROR, etc.), testset, passedTestCount, timeConsumedMillis, memoryConsumedBytes, creationTimeSeconds, original\_code. **No author or user\_id column** | Codeforces (May 2025\) | Node 3 trajectory mining; Node 2 clean C++20 SFT; Node 3 30% PASS calibration |
| nvidia/Nemotron-Competitive-Programming-v145 | 3,927,984 samples (\~178 GB JSONL): competitive\_coding\_cpp (932,012), competitive\_coding\_python (1,821,278), infinibyte (1,174,694) (cc-by-4.0) | Conversational messages (\[{"role": "user", "content": ...}\]) \+ metadata across 34,799 unique questions compiled from BAAI/TACO, codeparrot/apps, deepmind/code\_contests, open-r1/codeforces | Codeforces \+ **CodeChef** (e.g., EXMV, BESTOFTWO) \+ synthetic InfiniByte (Dec 2025; v2 Mar 2026\) | Node 1 blueprint distillation & CodeChef C++/Python synthesis |
| 4E-AI/Fidel-Competitive-coding46 | 10k–100k JSON records | Derived curation referencing nvidia/Nemotron-Competitive-Programming-v1 with CodeChef problem URLs | CodeChef \+ Codeforces | Compact CodeChef SFT seed (used for Fidel2.0-1B-GGUF) |
| PrimeIntellect/verifiable-coding-problems51 | 100k–1M Parquet records | Multi-language prompts (python, cpp, javascript, rust) and verifiable\_code schema adopted by open-r1; contains occasional null rows | Multi-platform | GRPO execution-verified rollouts (after dropping null entries) |
| deepmind/code\_contests54 | 13,328 training problems | Problem statements, ratings, public/private/generated test suites, AC and incorrect solutions | CodeChef, Codeforces, AtCoder, Aizu, HackerEarth (Feb 2022\) | **Primary CodeChef hidden test-case archive** (filtered for \>= 5 test cases) |

### **Filtering Queries and Quality-Control Pipeline**

Because open-r1/codeforces-cots is not pre-filtered for execution correctness (\~16% of Python traces fail public tests40) and PrimeIntellect/verifiable-coding-problems contains occasional null entries51, every training record requires deterministic deduplication and execution gating. Cross-dataset ingestion across open-r1/codeforces, deepmind/code\_contests, and nvidia/Nemotron-Competitive-Programming-v1 must also reconcile aliases (merging identical Div 1 / Div 2 problem pairs39) and normalized title/statement hashes so late-2024/2025 test problems cannot leak through Nemotron or PrimeIntellect.

Python  
from datasets import load\_dataset

\# 1\. Load verifiable Codeforces problems and anchor strict temporal leakage cutoff  
cf\_probs \= load\_dataset("open-r1/codeforces", "verifiable", split="train")  
cf\_test \= load\_dataset("open-r1/codeforces", "default", split="test") \# 468 late-2024/2025 problems

test\_ids \= set(cf\_test\["id"\])  
for al in cf\_test\["aliases"\]:  
 if al:  
 test\_ids.update(al)

min\_test\_ts \= min(x for x in cf\_test\["contest\_start"\] if x is not None)

def valid\_cf\_problem(row):  
 if row\["id"\] in test\_ids or any(a in test\_ids for a in (row\["aliases"\] or \[\])):  
 return False  
 if row\["contest\_start"\] is None or row\["contest\_start"\] \>= min\_test\_ts:  
 return False  
 if row\["interaction\_format"\] is not None: \# Exclude interactive problems for single-shot baseline  
 return False  
 if not row\["executable"\]:  
 return False  
 total\_tests \= len(row\["official\_tests"\] or \[\]) \+ (row\["generated\_tests"\] or 0\)  
 return total\_tests \>= 5 \# Enforce \>= 5 test cases per TinyLoRA-GRPO-Coder reliability rule

clean\_cf \= cf\_probs.filter(valid\_cf\_problem, num\_proc=8)

\# 2\. Extract CodeChef problems from deepmind/code\_contests dynamically by source label  
cc\_train \= load\_dataset("deepmind/code\_contests", split="train")  
source\_feature \= cc\_train.features.get("source")  
codechef\_label \= source\_feature.str2int("CODECHEF") if hasattr(source\_feature, "str2int") else None

def valid\_codechef(row):  
 is\_cc \= (row.get("source") \== codechef\_label) if codechef\_label is not None else ("codechef" in str(row).lower())  
 n\_tests \= sum(len(row.get(k, {}).get("input", \[\])) for k in ("public\_tests", "private\_tests", "generated\_tests"))  
 return is\_cc and n\_tests \>= 5

clean\_codechef \= cc\_train.filter(valid\_codechef, num\_proc=8)

\# 3\. Stream Nemotron C++ split (\~178 GB total dataset) to extract CodeChef-linked reasoning rows without OOM  
nemotron\_cpp \= load\_dataset(  
 "nvidia/Nemotron-Competitive-Programming-v1",  
 "competitive\_coding\_cpp",  
 split="train",  
 streaming=True,  
)  
codechef\_stream \= (row for row in nemotron\_cpp if "codechef.com" in str(row).lower())

For open-r1/codeforces-submissions, DuckDB queries the local 5.64 GB Parquet files directly, anchoring the temporal cutoff dynamically against the open-r1/codeforces test split:

SQL  
\-- Execute via DuckDB over local Parquet snapshots of open-r1/codeforces and open-r1/codeforces-submissions  
\-- Inspect DISTINCT programmingLanguage first to confirm exact Python/PyPy compiler tags in your shard  
WITH test\_cutoff AS (  
 SELECT MIN(contest\_start) AS min\_test\_start  
 FROM read\_parquet('open-r1-codeforces/default/test-\*.parquet')  
 WHERE contest\_start IS NOT NULL  
),  
clean\_problems AS (  
 SELECT id AS problem\_id, rating, time\_limit, memory\_limit  
 FROM read\_parquet('open-r1-codeforces/verifiable/train-\*.parquet'), test\_cutoff  
 WHERE executable \= true  
 AND interaction\_format IS NULL  
 AND rating BETWEEN 800 AND 1900  
 AND (len(official\_tests) \+ COALESCE(generated\_tests, 0)) \>= 5  
 AND contest\_start \< test\_cutoff.min\_test\_start  
)  
SELECT  
 s.submission\_id,  
 s.problem\_id,  
 s.programmingLanguage,  
 s.verdict,  
 s.passedTestCount,  
 s.timeConsumedMillis,  
 s.memoryConsumedBytes,  
 s.creationTimeSeconds,  
 s.source,  
 p.rating,  
 p.time\_limit,  
 p.memory\_limit  
FROM read\_parquet('open-r1-codeforces-submissions/data/\*.parquet') s  
INNER JOIN clean\_problems p ON s.problem\_id \= p.problem\_id  
WHERE (  
 s.programmingLanguage IN ('C++17 (GCC 9-64)', 'C++20 (GCC 11-64)', 'C++20 (GCC 13-64)')  
 OR s.programmingLanguage LIKE '%Python%'  
 OR s.programmingLanguage LIKE '%PyPy%'  
 )  
 AND s.verdict IN ('OK', 'TIME\_LIMIT\_EXCEEDED', 'WRONG\_ANSWER', 'MEMORY\_LIMIT\_EXCEEDED', 'RUNTIME\_ERROR')  
 AND length(s.source) BETWEEN 80 AND 12000;

### **Difficulty Curriculum and Mitigating Grandmaster Rating Overhang**

Although no controlled ablation in the retrieved literature defines a formal "2200+ rating poisoning" threshold by name, two empirical findings justify capping training difficulty well below Grandmaster (\>= 2200\) ratings at 0.5B–1.5B scale. First, DeepSeek-R1-Distill-Qwen-1.5B scores a 954 Codeforces rating baseline (and 7B reaches 1189\)13, placing 2200+ problems over 1,200 Elo points beyond native capability. Second, rSIM (*ACL 2026*) demonstrated that 0.5B RL policies suffer total reward collapse when tasks exceed capacity2, while NeurIPS 2025 curriculum research on 1.5B reasoning models showed that structured difficulty staging prevents degradation and outperforms flat DeepScaleR-1.5B (33.86%) training57. Exposing 0.5B or 1.5B models to 2200+ problems containing deep segment-tree lazy propagation or FFT proofs yields either near-zero GRPO advantage variance or brittle SFT memorization of ungrounded magic constants that degrades syntax discipline on solvable 800–1900 tasks.  
To align training complexity with sub-2B capacity and eliminate benchmark contamination, structure ingestion into three sequential stages:

> 1. **Phase 1 — Structural Discipline and Core Patterns (CF 800–1600 / CodeChef Div 4–Div 3\)**: Trains Node 1 on exact complexity budgeting (O(N) and O(N log N) vs. O(N^2)), Node 2 on fast I/O, 64-bit integer promotion (long long), greedy/sorting/prefix-sum/two-pointer templates, and diff patching, and Node 3 on basic edge-case bugs (N \= 1, 0, integer overflow) and quadratic TLE fixes.
> 2. **Phase 2 — Algorithmic Depth and Data Structures (CF 1600–1900 / CodeChef Div 2\)**: Introduces graph traversal, Dijkstra, DSU, binary search on the answer, combinatorics, and 1D/2D dynamic programming, strictly excluding \>= 2200 problems across Node 2 and Node 3 (and optionally allowing verified 1900–2100 teacher-distilled blueprints only in Node 1).
> 3. **Temporal Contest Splitting**: Enforces strict chronological separation aligned with open-r1/codeforces test (468 late-2024/early-2025 problems39) and LiveCodeBench out-of-distribution windows (e.g., post-July 202422), propagated across Nemotron, code\_contests, and PrimeIntellect via problem IDs, aliases, and 8-gram decontamination39.

### **Mining Submission Trajectories for the Critic**

A structural limitation of open-r1/codeforces-submissions shapes trajectory extraction: **the schema contains no author, handle, or user\_id column**44. Because the 12.7M records merge eight public sources without contestant identifiers44, pairing an arbitrary TLE or WA submission with an unrelated OK solution produces noisy, full-file rewrites rather than minimal diagnostic patches.  
To approximate same-author iterative repair trajectories (TLE \-\> AC, WA \-\> AC, MLE/RUNTIME\_ERROR \-\> AC) in the absence of user\_id, apply a **three-step engineered heuristic reconstruction** within identical (problem\_id, programmingLanguage) buckets:

> 1. **Comment Stripping and Lexical Tokenization**: Strip comments and whitespace, then tokenize source code while preserving identifier names and macro headers, which act as strong author-style fingerprints in competitive programming.
> 2. **Temporal-Window Similarity Matching**: For each failing submission (TIME\_LIMIT\_EXCEEDED, WRONG\_ANSWER, MEMORY\_LIMIT\_EXCEEDED, RUNTIME\_ERROR), scan chronologically subsequent OK submissions within a bounded contest/upsolve window (e.g., 0 \< dt \<= 3,600 s), requiring token Jaccard similarity J \>= 0.60 and compact unified diffs (\<= 60 diff lines).
> 3. **Physical Telemetry Verification**: Validate genuine resource improvement per transition class—requiring ac\_ms \<= 0.5 \* fail\_ms for TLE \-\> AC and ac\_kb \<= 0.7 \* fail\_kb for MLE \-\> AC using timeConsumedMillis and memoryConsumedBytes44—to discard noisy matches that happened to solve the problem via an unrelated rewrite.

Python  
import difflib  
import re  
from typing import Iterator, Dict, Any

TOKEN\_RE \= re.compile(r"\[A-Za-z\_\]\[A-Za-z0-9\_\]\*\|\\d+\|==\|\!=\|\<=\|\>=\|\<\<\|\>\>\|\[{}()\\\[\\\];,+\\-\*/%\<\>&\|^\~\!\]")

def strip\_comments\_and\_tokenize(code: str) \-\> set\[str\]:  
 code \= re.sub(r"//.\*?\$\|/\\\*.\*?\\\*/\|\#.\*?\$", "", code, flags=re.MULTILINE \| re.DOTALL)  
 return set(TOKEN\_RE.findall(code))

def mine\_author\_free\_trajectories(  
 submissions\_by\_problem\_lang: list\[Dict\[str, Any\]\],  
 max\_delta\_sec: int \= 3600,  
 min\_jaccard: float \= 0.60,  
) \-\> Iterator\[Dict\[str, Any\]\]:  
 ordered \= sorted(submissions\_by\_problem\_lang, key=lambda r: r\["creationTimeSeconds"\])  
 oks \= \[r for r in ordered if r\["verdict"\] \== "OK"\]  
 fails \= \[r for r in ordered if r\["verdict"\] in {  
 "TIME\_LIMIT\_EXCEEDED", "WRONG\_ANSWER", "MEMORY\_LIMIT\_EXCEEDED", "RUNTIME\_ERROR"  
 }\]

```
for fail in fails:  
    fail\_toks \= strip\_comments\_and\_tokenize(fail\["source"\])  
    if len(fail\_toks) \< 15:  
        continue  
    best\_match, best\_score \= None, 0.0  
    for ok in oks:  
        dt \= ok\["creationTimeSeconds"\] \- fail\["creationTimeSeconds"\]  
        if dt \<= 0:  
            continue  
        if dt \> max\_delta\_sec:  
            break  
        if fail\["verdict"\] \== "TIME\_LIMIT\_EXCEEDED" and ok\["timeConsumedMillis"\] \> 0.5 \* max(1, fail\["timeConsumedMillis"\]):  
            continue  
        if fail\["verdict"\] \== "MEMORY\_LIMIT\_EXCEEDED" and ok\["memoryConsumedBytes"\] \> 0.7 \* max(1, fail\["memoryConsumedBytes"\]):  
            continue  
        ok\_toks \= strip\_comments\_and\_tokenize(ok\["source"\])  
        jaccard \= len(fail\_toks & ok\_toks) / max(1, len(fail\_toks \| ok\_toks))  
        if jaccard \>= min\_jaccard and jaccard \> best\_score:  
            best\_score \= jaccard  
            best\_match \= ok

    if best\_match is not None:  
        patch \= "".join(difflib.unified\_diff(  
            fail\["source"\].splitlines(keepends=True),  
            best\_match\["source"\].splitlines(keepends=True),  
            fromfile="a/solution.cpp",  
            tofile="b/solution.cpp",  
            n=3,  
        ))  
        if 0 \< len(patch.splitlines()) \<= 60:  
            yield {  
                "problem\_id": fail\["problem\_id"\],  
                "fail\_verdict": fail\["verdict"\],  
                "fail\_code": fail\["source"\],  
                "ac\_code": best\_match\["source"\],  
                "unified\_diff": patch,  
                "jaccard\_similarity": round(best\_score, 4),  
                "fail\_ms": fail\["timeConsumedMillis"\],  
                "ac\_ms": best\_match\["timeConsumedMillis"\],  
                "fail\_kb": fail\["memoryConsumedBytes"\] // 1024,  
                "ac\_kb": best\_match\["memoryConsumedBytes"\] // 1024,  
            }
```

## **3\. Distilling and Fine-Tuning the Three Specialized Nodes with bf16 LoRA and GRPO**

### **Single-GPU 10 GB VRAM Workstation Feasibility Analysis**

Running true bf16 LoRA (rather than 4-bit QLoRA) for 0.5B–1.5B models on a single 10 GB Ampere GPU such as an RTX 3080 requires controlling Qwen2.5's 151,646-token vocabulary memory footprint25. Applying r=32, lora\_alpha=64 across all seven linear projections (q\_proj, k\_proj, v\_proj, o\_proj, gate\_proj, up\_proj, down\_proj) on Qwen2.5-Coder-1.5B-Instruct (28 layers, hidden=1536, intermediate=8960, 2 KV heads × 128 head dim \= 25624) adds 1,318,912 trainable parameters per layer (98,304 each for q/o, 57,344 each for k/v, 335,872 each for gate/up/down), totaling 36.93M LoRA parameters (\~2.40% of 1.54B).  
However, standard causal language modeling loss materializes a full \[batch\_size, seq\_len, vocab\_size\] logits tensor cast to fp32 during cross-entropy calculation. At batch\_size=2, max\_seq\_length=2048, vocab=151,646, that single fp32 tensor allocates 2 × 2048 × 151,646 × 4 bytes \= 2.48 GB (roughly double during backward gradient passes), pushing naive training into OOM on 10 GB VRAM. Activating **Unsloth's chunked cross-entropy** (or liger-kernel fused linear cross-entropy) together with gradient\_checkpointing="unsloth" and adamw\_8bit bypasses full logits allocation, keeping estimated peak bf16 LoRA VRAM safely within a 10 GB budget.

| Memory Component (10 GB RTX 3080\) | Node 1 / Node 3 (1.54B bf16 LoRA, r=32) | Node 2 (0.49B bf16 LoRA, r=32) | Node 2 GRPO (G=4, Adapter-Disabled Ref) |
| --- | --- | --- | --- |
| Base Model Weights (bf16, exact) | 3.08 GB (\~3.60 GB for 1.8B R1-Distill) | 0.98 GB | 0.98 GB (PEFT adapter toggle avoids 2nd model copy) |
| LoRA \+ Grads \+ adamw\_8bit (est.) | \~0.30 GB (36.93M trainable params) | Proportionally smaller (\~1/3 of 1.5B LoRA) | Proportionally smaller |
| Activations \+ KV (seq=2048, bs=2, est.) | \~1.8–2.4 GB (with gradient checkpointing) | \~0.9–1.2 GB | \~2.2–3.0 GB (4 rollout completions \+ logprobs) |
| Chunked Cross-Entropy Loss (est.) | \~0.3 GB *(vs. 2.48 GB forward / \~4.9 GB peak naive fp32 logits)* | \~0.3 GB | \~0.3 GB |
| **Estimated Peak VRAM** | **\~5.5–6.3 GB** *(fits 10 GB safely)* | **\~2.3–2.6 GB** | **\~3.6–4.4 GB** *(fits 10 GB safely)* |

(Estimation caveat: Base weight sizes and the 2.48 GB naive fp32 logits buffer are exact arithmetic from verified parameter and vocabulary counts23; activation, optimizer, and chunked cross-entropy figures are analytical workstation estimates rather than measured RTX 3080 profiling logs.)  
For trl GRPOTrainer, Node 2 (0.49B) accommodates num\_generations=4–6 comfortably inside 10 GB VRAM. Running 1.5B GRPO for Node 1 or Node 3 on 10 GB requires disabling vLLM colocation (using native Hugging Face generation capped at max\_completion\_length=384), setting per\_device\_train\_batch\_size=1, num\_generations=4, and omitting a duplicate ref\_model so PEFT computes reference log-probabilities by temporarily disabling the LoRA adapter.

### **Node 1: System 2 Algorithmic Planner**

Node 1 can initialize from DeepSeek-R1-Distill-Qwen-1.5B (higher 954 Codeforces baseline13, SFT-distilled over 800K curated samples13) or Qwen2.5-Coder-1.5B-Instruct (allowing shared-backbone multi-LoRA Option B deployment with Node 324). Because raw DeepSeek-R1 reasoning traces in open-r1/codeforces-cots frequently span several thousand tokens40, emitting uncompressed traces on CPU creates unacceptable latency: reducing a 4,000-token reasoning trace to at most 250 tokens cuts generation volume by \>= 16×, shrinking decode duration from 100 s at 40 tok/s down to roughly 6.25 s (or \~3.1 s at 80 tok/s).  
Recent 2025–2026 CoT-compression works—including TokenSkip (supervising on shortened valid traces58), CoT-Valve (length-compressible tuning evaluated on DeepSeek-R1-Distill-Qwen-1.5B60), ADR (49.5%–59.3% length reduction with up to 6.1% gain62), ARM2 (\>70% token reduction vs. GRPO62), and adaptive length-reward shaping (LASER-D, HAPO, AutoThink13)—show that pruned traces preserve algorithmic accuracy. Build Node 1 in three sequential stages:

> 1. **Execution-Filtered Extraction**: Filter open-r1/codeforces-cots (solutions\_decontaminated, solutions\_w\_editorials\_decontaminated) and CodeChef records from Nemotron-Competitive-Programming-v140 strictly by running each extracted solution against \>= 5 verified tests55.
> 2. **Blueprint Distillation (\<250 Tokens)**: Condense each verified trace and editorial into a compact JSON schema containing analysis (short invariant/proof summary), paradigm, data\_structures, time\_complexity, space\_complexity, use\_int64 (true whenever intermediate products can exceed 2 × 10^9), edge\_cases, and steps.
> 3. **bf16 LoRA SFT and Length-Penalized GRPO**: Train Node 1 via bf16 LoRA SFT (r=32, lora\_alpha=64, effective batch size 16 matching rSIM2, heuristic LoRA LR 1e-4—roughly 10× rSIM's 1e-5 full-parameter LR2—cosine decay, 2 epochs), optionally followed by ARM2/LASER-D-style GRPO rewarding \+1.0 when downstream Node 2 execution achieves AC minus a linear over-budget token penalty past 200 tokens.

### **Node 2: System 1 Fast Coder and Diff Patcher**

Node 2 (Qwen2.5-Coder-0.5B-Instruct) executes pure CoT-free translation trained on an interleaved mix of full generation and localized patching (e.g., a 60% / 40% curriculum allocation). In the Plan-to-Code task, the model maps (Problem I/O Summary \+ Node 1 JSON Blueprint) directly to clean C++20 or Python sourced from open-r1/codeforces-submissions (selected\_accepted) and Nemotron44, enforcing zero conversational preamble, ios::sync\_with\_stdio(false); cin.tie(nullptr);, explicit long long aliases, and '\\n' instead of endl. In the Error-to-Unified-Diff task modeled on SRI-Coder15, Node 2 maps (Broken Code \+ Compiler stderr OR Sample Diff \+ Node 3 Patch Directive) to a clean unified diff patch, sharpened with DPO pairs contrasting AC-producing diffs against patches that fail patch \-p1, cause compile errors, or alter unrelated template boilerplate. Because rSIM proved standalone RL collapses on Qwen2.5-0.5B2, any optional Node 2 GRPO stage must condition exclusively on frozen Node 1 blueprints, scoring 1.0 for full test-suite AC, 0.2 for clean g++ \-O2 \-std=c++20 compilation, and 0.0 for unapplicable diffs or conversational text.

### **Node 3: Jev Critic and Complexity Optimizer**

A diagnostic critic trained solely on failing submissions develops **critic paranoia**, falsely rejecting optimal O(N log N) AC solutions and rewriting passing code into broken variants. To anchor precision, train Node 3 (Qwen2.5-Coder-1.5B-Instruct) with SFT plus REDI-style positive/negative contrastive DPO13 on a balanced mixture containing **30% optimal AC submissions labeled {"status": "PASS", "escalate\_to\_planner": false, "patch\_directive": null}** alongside trajectory-mined repairs (allocated as a 30% PASS / 35% OPTIMIZE / 35% BUG engineering mix). Optimal AC calibration examples pair selected\_accepted code with healthy sandbox metrics (for example, CPU time \<= 35% of the problem limit and RSS \<= 50% of the memory cap) so the critic learns when not to intervene. Trajectory-mined TLE/MLE \-\> AC repairs train "status": "OPTIMIZE"—setting "escalate\_to\_planner": true when asymptotic complexity must improve or false for constant-factor fixes like I/O buffering—while WA/RE/CE \-\> AC pairs train "status": "BUG" with minimal targeted line directives for Node 2\.

Python  
\# Runnable bf16 LoRA SFT configuration for 10 GB RTX 3080 (Unsloth \+ TRL)  
from unsloth import FastLanguageModel  
from trl import SFTTrainer, SFTConfig  
import torch

model, tokenizer \= FastLanguageModel.from\_pretrained(  
 model\_name="Qwen/Qwen2.5-Coder-1.5B-Instruct", \# swap to 0.5B-Instruct for Node 2  
 max\_seq\_length=2048,  
 dtype=torch.bfloat16,  
 load\_in\_4bit=False, \# True bf16 LoRA (1.54B base \= 3.08 GB VRAM)  
)

model \= FastLanguageModel.get\_peft\_model(  
 model,  
 r=32,  
 lora\_alpha=64,  
 lora\_dropout=0.0,  
 target\_modules=\["q\_proj", "k\_proj", "v\_proj", "o\_proj", "gate\_proj", "up\_proj", "down\_proj"\],  
 bias="none",  
 use\_gradient\_checkpointing="unsloth", \# Activates chunked cross-entropy to avoid 151k-vocab fp32 logit spike  
 random\_state=42,  
)

sft\_args \= SFTConfig(  
 output\_dir="./checkpoints/node3-jev-critic-bf16",  
 per\_device\_train\_batch\_size=2,  
 gradient\_accumulation\_steps=8, \# Effective batch size 16  
 learning\_rate=1e-4, \# Standard LoRA LR heuristic (use 2e-4 for 0.5B Node 2\)  
 lr\_scheduler\_type="cosine",  
 warmup\_ratio=0.05,  
 num\_train\_epochs=2,  
 bf16=True,  
 optim="adamw\_8bit",  
 max\_seq\_length=2048,  
 packing=True,  
 logging\_steps=20,  
 save\_strategy="epoch",  
)

trainer \= SFTTrainer(  
 model=model,  
 tokenizer=tokenizer,  
 train\_dataset=balanced\_critic\_dataset, \# 30% PASS / 35% OPTIMIZE / 35% BUG  
 dataset\_text\_field="text",  
 args=sft\_args,  
)  
trainer.train()

## **4\. Grammar-Constrained Decoding and Deterministic Evaluation Harness**

### **Resolving the Structured-Output Reasoning Trade-Off**

In *"Let Me Speak Freely?"* (*EMNLP 2024*), Tam et al. showed that strict format constraints—particularly rigid JSON schemas—can degrade LLM reasoning accuracy when models are forced directly into constrained structural slots without free-form deliberation63. At the same time, work by Geng et al., Raspanti et al., and 2026 deployment evaluations conclude that llama.cpp GBNF grammar enforcement is indispensable for small models when deterministic downstream parsing is required, freeing sub-2B capacity from formatting errors67. Placing a bounded free-text "analysis" field as the very first JSON key in both the Planner and Critic schemas reconciles this tension, preserving token-level workspace for constraint checks before locking downstream fields67.

### **Complete llama.cpp GBNF Grammars**

**1\. Node 1 Planner JSON GBNF (planner.gbnf)**

Code snippet  
root ::= "{" ws  
 "\\"analysis\\":" ws bounded-string "," ws  
 "\\"paradigm\\":" ws paradigm-enum "," ws  
 "\\"data\_structures\\":" ws string-array "," ws  
 "\\"time\_complexity\\":" ws complexity-string "," ws  
 "\\"space\_complexity\\":" ws complexity-string "," ws  
 "\\"use\_int64\\":" ws boolean "," ws  
 "\\"edge\_cases\\":" ws string-array "," ws  
 "\\"steps\\":" ws string-array  
ws "}"

paradigm-enum ::= "\\"greedy\\"" \| "\\"dp\\"" \| "\\"graph\\"" \| "\\"binary\_search\\"" \| "\\"math\_number\_theory\\"" \| "\\"two\_pointers\\"" \| "\\"data\_structures\\"" \| "\\"brute\_force\\"" \| "\\"constructive\\""  
complexity-string ::= "\\"O(" \[A-Za-z0-9 \*+^()\_/-\]{1,32} ")\\""  
boolean ::= "true" \| "false"  
string-array ::= "\[" ws (string ("," ws string){0,6})? ws "\]"  
bounded-string ::= "\\"" char{10,700} "\\""  
string ::= "\\"" char{1,160} "\\""  
char ::= \[^"\\\\\\x00-\\x1F\] \| "\\\\" (\["\\\\/bfnrt\] \| "u" \[0-9a-fA-F\]{4})  
ws ::= \[ \\t\\n\\r\]\*

**2\. Node 2 Coder Unified-Diff Patch GBNF (unified\_diff.gbnf) and Raw C++20 Prefix Lock (cpp\_raw.gbnf)**

Code snippet  
\# unified\_diff.gbnf: enforces valid unified diff patch with \>= 1 hunk  
root ::= "--- a/solution." ext "\\n" "+++ b/solution." ext "\\n" hunk+  
ext ::= "cpp" \| "py"  
hunk ::= "@@ \-" digits "," digits " \+" digits "," digits " @@" \[^\\n\]\* "\\n" hunk-line+  
hunk-line ::= (" " \| "+" \| "-") \[^\\n\]\* "\\n"  
digits ::= \[0-9\]+

Code snippet  
\# cpp\_raw.gbnf: forces C++20 preprocessor start and forbids backticks to block markdown fences  
root ::= "\#include" \[^\\x60\]+

*(For Python generation in Node 2, replace the \#include prefix with an optional import/from/def or non-backtick line prefix so \[^\\x60\]+ similarly suppresses Markdown code fences.)*  
**3\. Node 3 Jev Critic JSON Verdict GBNF (critic.gbnf)**

Code snippet  
root ::= "{" ws  
 "\\"analysis\\":" ws bounded-string "," ws  
 "\\"status\\":" ws status-enum "," ws  
 "\\"bottleneck\_class\\":" ws bottleneck-enum "," ws  
 "\\"escalate\_to\_planner\\":" ws boolean "," ws  
 "\\"target\_lines\\":" ws nullable-string "," ws  
 "\\"patch\_directive\\":" ws nullable-string  
ws "}"

status-enum ::= "\\"PASS\\"" \| "\\"OPTIMIZE\\"" \| "\\"BUG\\""  
bottleneck-enum ::= "\\"NONE\\"" \| "\\"COMPLEXITY\_TLE\\"" \| "\\"IO\_OVERHEAD\\"" \| "\\"MEMORY\_LIMIT\\"" \| "\\"INT\_OVERFLOW\\"" \| "\\"EDGE\_CASE\_WA\\"" \| "\\"OUT\_OF\_BOUNDS\_RE\\"" \| "\\"SYNTAX\_CE\\""  
boolean ::= "true" \| "false"  
nullable-string ::= "null" \| "\\"" char{1,320} "\\""  
bounded-string ::= "\\"" char{10,600} "\\""  
char ::= \[^"\\\\\\x00-\\x1F\] \| "\\\\" (\["\\\\/bfnrt\] \| "u" \[0-9a-fA-F\]{4})  
ws ::= \[ \\t\\n\\r\]\*

### **Property-Based Fuzz Testing and Differential Oracle Verification**

Public CodeChef pages expose only a handful of sample tests, and open-r1/codeforces truncates official test strings above 400 characters whenever official\_tests\_complete is false39. Evaluating candidate solutions solely on sample inputs therefore produces severe false-positive AC rates. Replicating the verification methodology established by open-r1/codeforces and open-r1/codeforces-cots (test\_input\_generator and checker\_interactor subsets) closes this gap locally39:

> 1. **Adversarial and Worst-Case Input Synthesis**: Using test\_input\_generator patterns or deterministic Python fuzzers, generate test suites covering minimal boundaries (N \= 1, K \= 0, uniform arrays, sorted/reverse-sorted permutations), graph pathologies (disconnected components, star trees, line chains of length N), 64-bit arithmetic traps (values near 10^9 overflowing 2^31 \- 1), and **maximum-constraint stress cases (N \= N\_max)** to expose real TLE and MLE failures invisible on small samples.
> 2. **Differential Multi-Oracle Consensus**: When hidden outputs are unavailable (as on unarchived CodeChef problems), execute \>= 2 independent verified human AC reference solutions sourced from deepmind/code\_contests or Nemotron against every fuzzed input inside the local sandbox39. Following open-r1/codeforces curation rules, discard or escalate to a custom checker any input where verified AC solutions produce differing strings39.
> 3. **Custom Multi-Answer Checkers**: Because roughly 30% of competitive programming problems permit multiple valid outputs39, non-unique problems require a problem-specific validator matching the open-r1/codeforces interface (python checker.py input.txt correct\_output.txt solution\_output.txt), emitting a 0–1 or 0–100 verdict score to stdout39.

## **5\. Sandboxed Local Judge and CPU Telemetry Extraction**

### **Linux Sandboxing Primitives for Untrusted Sub-2B Code**

Compiling and executing unvetted C++20 and Python solutions inside an agent loop demands kernel-level containment against fork bombs, infinite recursion, memory exhaustion, and filesystem or network escapes. Heavy container runtimes penalize high-frequency agent testing: in 2026 benchmarks on a Ryzen 5 5500U workstation (Linux 6.18), Docker startup latency measured roughly 44× slower than bare-metal Sandlock (\~5 ms Landlock \+ seccomp-bpf overhead vs. hundreds of milliseconds for container setup)70. While no controlled microbenchmark in the retrieved sources directly compares nsjail, bubblewrap, and firejail startup overhead side-by-side, their security boundaries and resource-control primitives differ substantially8.

| Sandbox Engine | Isolation Primitives | Privilege & Attack Surface | Resource Caps (CPU / RSS) | 2026 Stability & Competitive Judge Fit |
| --- | --- | --- | --- | --- |
| **nsjail (v3.6+)**8 | 8 Linux namespaces (UTS, mount, PID, IPC, net, user, cgroup, time), chroot/pivot\_root, Kafel seccomp-bpf policies | Unprivileged user namespaces (cgroup features require root or delegated cgroup v2) | Native rlimits (RLIMIT\_AS, RLIMIT\_CPU, RLIMIT\_NPROC) \+ **native cgroup v1/v2 memory & PID caps** | **Primary local judge**. Avoid nsjail 3.5 (flagged BROKEN over non-additive seccomp\_string config rules); deploy **nsjail 3.6** (Mar 18, 2026\)9. |
| **bubblewrap (bwrap)**8 | Unprivileged user/mount/PID/IPC/net namespaces (--unshare-all, \--ro-bind, \--die-with-parent) | Minimal attack surface; **no SUID root** | Lacks native cgroup/rlimit controller; pair with prlimit, systemd-run \--scope, and external \--seccomp FD | Recommended unprivileged fallback when nsjail is unavailable; supported natively by llm-sandbox8. |
| **firejail**70 | Linux namespaces, default /etc/firejail seccomp and AppArmor profiles | **Relies on a SUID-root binary**, historically prone to privilege escalation CVEs71 | Basic \--rlimit-\* flags | **Avoid** for high-frequency untrusted AI code execution due to expanded SUID attack surface73. |
| **seccomp-bpf / Sandlock**70 | Kernel BPF syscall allowlist \+ Landlock (Sandlock adds pidfd\_getfd live notification) | Unprivileged (Sandlock adds \~5 ms startup overhead70) | Requires external rlimit or watchdog for strict RSS limits | Strong syscall-filtering layer (--seccomp in bwrap or native Sandlock pipelines)70. |

### **Dual-Stage Compilation/Execution and CPU Telemetry Pipeline**

Because g++ \-O2 \-std=c++20 itself can suffer compile-time bombs or malicious \#include path traversal, compilation and binary execution run inside **separate nsjail 3.6 invocations** mounted inside an empty \$JAIL\_ROOT (never \--chroot /, which would expose host paths; nsjail isolates the network namespace by default unless \--disable\_clone\_newnet is passed). Pinning judge subprocesses via taskset to CPU E-cores prevents compilation memory traffic from displacing P-core llama-server weights.

Bash  
\# Stage 1: Sandboxed C++20 compilation inside isolated empty JAIL\_ROOT (network namespace isolated by default)  
nsjail \--mode o \--quiet \\  
 \--chroot "\$JAIL\_ROOT" \\  
 \--bindmount\_ro /usr:/usr \--bindmount\_ro /lib:/lib \--bindmount\_ro /lib64:/lib64 \--bindmount\_ro /bin:/bin \\  
 \--bindmount "\$WORKDIR":/work \--cwd /work \\  
 \--time\_limit 10 \--rlimit\_as 1024 \--rlimit\_cpu 10 \--rlimit\_fsize 32 \\  
 \-- /usr/bin/g++ \-O2 \-std=c++20 \-pipe \-Wall \-Wextra /work/solution.cpp \-o /work/sol

\# Stage 2: Competitive-judge execution under strict CodeChef limits (requires nsjail \>= 3.6)  
nsjail \--mode o \--quiet \\  
 \--chroot "\$JAIL\_ROOT" \\  
 \--bindmount\_ro /lib:/lib \--bindmount\_ro /lib64:/lib64 \\  
 \--bindmount\_ro "\$WORKDIR":/work \--cwd /work \\  
 \--time\_limit 2 \\  
 \--rlimit\_cpu 2 \--rlimit\_as 512 \--rlimit\_stack 256 \--rlimit\_nproc 1 \--rlimit\_fsize 16 \\  
 \--cgroup\_mem\_max \$((256 \* 1024 \* 1024)) \--cgroup\_pids\_max 1 \\  
 \--seccomp\_string "ALLOW { read, write, exit, exit\_group, brk, mmap, munmap, mprotect, fstat, newfstatat, lseek, getrandom, clock\_gettime, rt\_sigaction, rt\_sigreturn, arch\_prctl, set\_tid\_address, set\_robust\_list, rseq, execve } DEFAULT KILL" \\  
 \-- /work/sol \< "\$TEST\_INPUT" \> "\$WORK\_OUT" 2\> "\$WORK\_ERR"

After termination, the Python orchestrator collects os.wait4 resource statistics (ru\_utime, ru\_stime, ru\_maxrss), wall-clock nanoseconds (time.perf\_counter\_ns()), cgroup v2 memory.peak RSS, exit signals (SIGXCPU / timeout \-\> TLE, OOM 137 \-\> MLE, SIGSEGV/SIGABRT \-\> RE), and a truncated diff preview. Formatting this physical telemetry directly into Node 3's prompt grounds the Critic in real hardware measurements (illustrative payload schema below):

JSON  
{  
 "verdict": "TLE",  
 "wall\_time\_ms": 2004,  
 "cpu\_time\_ms": 1998,  
 "time\_limit\_ms": 1000,  
 "peak\_rss\_mb": 42.6,  
 "memory\_limit\_mb": 256,  
 "passed\_tests": 4,  
 "total\_tests": 18,  
 "failing\_input\_preview": "100000 100000\\n1 2 3 4 5 ...",  
 "expected\_vs\_actual\_diff": null,  
 "compiler\_stderr": null  
}

## **6\. Post-Training Quantization with GGUF and imatrix for CPU Inference**

### **Selecting Quantization Formats Across Sub-2B Roles**

At 0.5B to 1.5B parameters, quantization noise hits sparse syntactic tokens disproportionately: slight logit perturbations between ; and : or { and } break C++20 compilation or shift GBNF-constrained decoding trajectories. In llama.cpp, K-quants (Q4\_K\_M, Q5\_K\_M) distribute higher bit precision to sensitive attention value/output and feed-forward down-projection blocks while quantizing per-block constants74, whereas IQ4\_XS applies non-linear importance-matrix (imatrix) quantization75. An independent hardware guide (sesamedisk.com) reports that IQ4\_XS saves roughly 400 MB compared to Q4\_K\_M at nearly identical perplexity on multi-billion-parameter weights (at 1.5B scale, an \~0.5 bit-per-weight difference corresponds proportionally to roughly \~0.09–0.1 GB saved)75.  
It must be stated plainly that **no controlled benchmark in the retrieved 2024–2026 literature measures C++20 bracket/semicolon syntax error rates, GBNF schema failures, or AVX2 vs. AVX-512 CPU throughput across Q4\_K\_M, Q5\_K\_M, and IQ4\_XS on 0.5B–1.5B models**. Role assignments below therefore synthesize llama.cpp quantization mechanics and verified GGUF footprints6:

| GGUF Format | Weight Size (0.5B / 1.5B) | Quantization Properties | Recommended Node Assignment |
| --- | --- | --- | --- |
| **Q5\_K\_M** | \~0.44 GB (est.) / 1.29 GB (R1-Distill-1.5B verified33) | Mixed 5-bit/6-bit K-quant with bit-shift/mask unpacking74; preserves sharp logits on tight syntax distributions. | **Recommended for Node 2 (0.5B Coder/Patcher)** where tiny parameter headroom makes 4-bit brittle; preferred for Node 1/3 when \>= 2.8 GB RAM is available. |
| **Q4\_K\_M** | 398 MB (Ollama 0.5B34) / \~0.9–1.05 GB (est. 1.5B) | Mixed 4-bit/6-bit K-quant74; cuts memory-bandwidth bytes per token vs. Q5\_K\_M with straightforward SIMD unpacking. | **Default for Node 1 (1.5B Planner) and Node 3 (1.5B Critic)** inside an \~2.5 GB total RAM budget. |
| **IQ4\_XS** | \~0.35 GB (est.) / \~0.85–0.95 GB (est.) | Importance-weighted non-linear quantization75; smallest 4-bit-class footprint at near-Q4\_K\_M perplexity75. | Reserve for tight \< 2.2 GB RAM ceilings after benchmarking local AVX2 or AVX-512 decode speed via llama-bench. |

Across all three quantized nodes, keep KV-cache precision at **q8\_0** on post-2026-04-01 Hadamard-rotated llama.cpp builds and avoid q4\_0 KV caching on code-repair or reasoning tasks6.

### **Competitive-Programming imatrix Calibration Workflow**

Calibrating llama-imatrix solely on generic prose such as WikiText-2 leaves activation statistics unadapted to C++20 operators (\<\<=, \-\>, ::), unified diff headers (@@ \-12,4 \+12,6 @@), and JSON escape sequences. Because llama-imatrix accumulates squared activation magnitudes across linear projections during forward passes, building calibration\_cp.txt from balanced thirds of Node 1 JSON blueprints, verified AC C++20/Python solutions plus unified diffs, and Node 3 diagnostic verdicts focuses quantization precision on competitive programming syntax and schema tokens (for example, 150–250 chunks totaling roughly 200K–300K tokens as an engineering calibration target):

Bash  
\# 1\. Merge bf16 LoRA adapter into base weights (PEFT merge\_and\_unload), then convert to F16 GGUF  
python llama.cpp/convert\_hf\_to\_gguf.py \\  
 ./merged\_models/node1-planner-1.5b \\  
 \--outtype f16 \\  
 \--outfile ./gguf/node1-planner-1.5b-f16.gguf

\# 2\. Compute competitive-programming-specific Importance Matrix (imatrix)  
./llama.cpp/build/bin/llama-imatrix \\  
 \-m ./gguf/node1-planner-1.5b-f16.gguf \\  
 \-f ./calibration/calibration\_cp.txt \\  
 \-o ./gguf/node1-planner-1.5b.imatrix \\  
 \-c 2048 \--chunks 200 \-t 8

\# 3\. Quantize with domain imatrix (Q5\_K\_M for 0.5B Coder; Q4\_K\_M or IQ4\_XS for 1.5B Planner/Critic)  
./llama.cpp/build/bin/llama-quantize \\  
 \--imatrix ./gguf/node1-planner-1.5b.imatrix \\  
 ./gguf/node1-planner-1.5b-f16.gguf \\  
 ./gguf/node1-planner-1.5b-Q4\_K\_M.gguf \\  
 Q4\_K\_M

\# 4\. Validate quantized perplexity and q8\_0 KV cache integrity against F16 on held-out CP traces  
./llama.cpp/build/bin/llama-perplexity \\  
 \-m ./gguf/node1-planner-1.5b-Q4\_K\_M.gguf \\  
 \-f ./calibration/heldout\_cp\_val.txt \\  
 \-c 2048 \-ctk q8\_0 \-ctv q8\_0

Before promoting quantized checkpoints into the agentic judge loop, verify three deterministic regression gates against the f16 GGUF baseline on a held-out validation slice6: 100% first-pass JSON parseability under GBNF, clean g++ \-O2 \-std=c++20 compilation parity (for example, retaining within \<= 1.5% of f16 compile rate as a production acceptance bar), and end-to-end Pass@1 verification on 800–1900 problems from the temporal open-r1/codeforces test split inside nsjail 3.6. By pairing temporal-split curricula, \<250-token blueprint distillation, 30% PASS-calibrated Critic preferences, GBNF contracts, P-core affinity masking, and domain-calibrated imatrix quantization, a CPU-resident sub-2B hierarchy turns iterative sandbox verification into competitive programming accuracy at zero marginal inference cost.

#### **Works cited**

> 1. A Survey of Reasoning Large Language Models \- ResearchGate, [https\://www\.researchgate.net/publication/397976557\_From\_System\_1\_to\_System\_2\_A\_Survey\_of\_Reasoning\_Large\_Language\_Models](https://www.researchgate.net/publication/397976557_From_System_1_to_System_2_A_Survey_of_Reasoning_Large_Language_Models)
> 2. rSIM: Incentivizing Reasoning Capabilities of LLMs via Reinforced, [https\://aclanthology.org/2026.acl-long.2054.pdf](https://aclanthology.org/2026.acl-long.2054.pdf)
> 3. \[2506.21734\] Hierarchical Reasoning Model \- arXiv, [https\://arxiv.org/abs/2506.21734](https://arxiv.org/abs/2506.21734)
> 4. Hierarchical Reasoning Model \- arXiv, [https\://arxiv.org/html/2506.21734v1](https://arxiv.org/html/2506.21734v1)
> 5. How to swap LoRA during a post request to llama-server? \#8849, [https\://github.com/ggml-org/llama.cpp/discussions/8849](https://github.com/ggml-org/llama.cpp/discussions/8849)
> 6. KV Cache Quantization: What FP8 and q8\_0 Cost in Accuracy, [https\://particula.tech/blog/kv-cache-quantization-accuracy-loss-benchmarks](https://particula.tech/blog/kv-cache-quantization-accuracy-loss-benchmarks)
> 7. nikola/llama\_cpp \- llama\_cpp \- Gitea: Git with a cup of tea, [https\://git.petrovv.com/nikola/llama\_cpp/commits/commit/7c7f3b7f435f41f2508e0e3010f0013cd8335156/examples/llama-bench](https://git.petrovv.com/nikola/llama_cpp/commits/commit/7c7f3b7f435f41f2508e0e3010f0013cd8335156/examples/llama-bench)
> 8. LLM Sandbox Goes Containerless: Bubblewrap and NsJail Backends, [https\://blog.duy.dev/llm-sandbox-containerless-bubblewrap-nsjail/](https://blog.duy.dev/llm-sandbox-containerless-bubblewrap-nsjail/)
> 9. nsjail review: features, isolation & pricing (2026) \- Agent Sandboxes, [https\://agentsandboxes.org/sandboxes/nsjail/](https://agentsandboxes.org/sandboxes/nsjail/)
> 10. deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \- Hugging Face, [https\://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
> 11. DeepSeek-R1 \- GitHub, [https\://github.com/deepseek-ai/deepseek-r1](https://github.com/deepseek-ai/deepseek-r1)
> 12. DeepSeek-R1-Distill-Qwen-1.5B: The best small-sized LLM? \- Medium, [https\://medium.com/data-science-in-your-pocket/deepseek-r1-distill-qwen-1-5b-the-best-small-sized-llm-14eee304d94b](https://medium.com/data-science-in-your-pocket/deepseek-r1-distill-qwen-1-5b-the-best-small-sized-llm-14eee304d94b)
> 13. DeepSeek-R1-Distill-Qwen-1.5B: Efficient Reasoning \- Emergent Mind, [https\://www\.emergentmind.com/topics/deepseek-r1-distill-qwen-1-5b-model](https://www.emergentmind.com/topics/deepseek-r1-distill-qwen-1-5b-model)
> 14. EFFICIENT LLM COLLABORATION VIA PLANNING \- OpenReview, [https\://openreview.net/pdf?id=HrB7sYLd5N](https://openreview.net/pdf?id=HrB7sYLd5N)
> 15. From Completion to Editing: Unlocking Context-Aware Code Infilling, [https\://aclanthology.org/2026.acl-long.361.pdf](https://aclanthology.org/2026.acl-long.361.pdf)
> 16. From Completion to Editing: Unlocking Context-Aware Code Infilling, [https\://arxiv.org/html/2601.13384v1](https://arxiv.org/html/2601.13384v1)
> 17. Towards Scalable and Resource- Conscious ... \- Preprints.org, [https\://www\.preprints.org/manuscript/202505.1517/v1/download](https://www.preprints.org/manuscript/202505.1517/v1/download)
> 18. Foundations of Large Language Models \- arXiv, [https\://arxiv.org/pdf/2501.09223](https://arxiv.org/pdf/2501.09223)
> 19. Improving LLM Code Generation via Requirement-Aware ... \- arXiv, [https\://arxiv.org/html/2605.00433v1](https://arxiv.org/html/2605.00433v1)
> 20. Daily Papers \- Hugging Face, [https\://huggingface.co/papers?q=reasoning%20context](https://huggingface.co/papers?q=reasoning+context)
> 21. Experiments with reasoning models, training techniques, papers, [https\://github.com/benjaminzwhite/reasoning-models](https://github.com/benjaminzwhite/reasoning-models)
> 22. Qwen2.5-Coder Series: Powerful, Diverse, Practical., [https\://qwen.ai/blog?id=qwen2.5-coder-family](https://qwen.ai/blog?id=qwen2.5-coder-family)
> 23. Qwen/Qwen2.5-Coder-0.5 B-Instruct-GPTQ-Int4 \- ModelScope, [https\://www\.modelscope.cn/models/Qwen/Qwen2.5-Coder-0.5B-Instruct-GPTQ-Int4](https://www.modelscope.cn/models/Qwen/Qwen2.5-Coder-0.5B-Instruct-GPTQ-Int4)
> 24. Qwen/Qwen2.5-Coder-1.5B-Instruct \- Hugging Face, [https\://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
> 25. Qwen2.5-Coder Technical Report \- arXiv, [https\://arxiv.org/html/2409.12186v1](https://arxiv.org/html/2409.12186v1)
> 26. Qwen2.5-Coder Technical Report \- arXiv, [https\://arxiv.org/html/2409.12186v3](https://arxiv.org/html/2409.12186v3)
> 27. Qwen2.5-Coder 1.5B Instruct API & Playground \- Fireworks AI, [https\://fireworks.ai/models/fireworks/qwen2p5-coder-1p5b-instruct](https://fireworks.ai/models/fireworks/qwen2p5-coder-1p5b-instruct)
> 28. DeepSeek R1 Distill Qwen 1.5B Benchmarks, Pricing & Context, [https\://llm-stats.com/models/deepseek-r1-distill-qwen-1.5b](https://llm-stats.com/models/deepseek-r1-distill-qwen-1.5b)
> 29. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs ... \- arXiv, [https\://arxiv.org/html/2501.12948v1](https://arxiv.org/html/2501.12948v1)
> 30. How to set llama.cpp server LoRA adapter scales \- Simplified Guide, [https\://www\.simplified.guide/llama-cpp/server-set-lora-adapter](https://www.simplified.guide/llama-cpp/server-set-lora-adapter)
> 31. Steering Vision-Language Models at Runtime with LoRA Adapters, [https\://medium.com/openvino-toolkit/steering-vision-language-models-at-runtime-with-lora-adapters-and-openvino-genai-9cfd41a1c61a](https://medium.com/openvino-toolkit/steering-vision-language-models-at-runtime-with-lora-adapters-and-openvino-genai-9cfd41a1c61a)
> 32. KV Cache and KV Caching Explained: Memory Math for LLM Inference, [https\://blog.prompt20.com/posts/kv-cache/](https://blog.prompt20.com/posts/kv-cache/)
> 33. Getting Started with DeepSeek-R1-Distill-Qwen-1.5B \- SecondState.io, [https\://www\.secondstate.io/articles/deepseek-r1-distill-qwen/](https://www.secondstate.io/articles/deepseek-r1-distill-qwen/)
> 34. qwen2.5-coder:0.5b-instruct \- Ollama, [https\://ollama.com/library/qwen2.5-coder:0.5b-instruct](https://ollama.com/library/qwen2.5-coder:0.5b-instruct)
> 35. Ollama vs llama.cpp vs vLLM: choosing a local LLM server \- Dayfing, [https\://dayfing.com/writing/local-llm-inference-servers/](https://dayfing.com/writing/local-llm-inference-servers/)
> 36. UC Berkeley \- eScholarship.org, [https\://escholarship.org/content/qt853442rj/qt853442rj.pdf](https://escholarship.org/content/qt853442rj/qt853442rj.pdf)
> 37. open-r1/codeforces · Datasets at Hugging Face, [https\://huggingface.co/datasets/open-r1/codeforces/viewer/default/train](https://huggingface.co/datasets/open-r1/codeforces/viewer/default/train)
> 38. 📊 CodeForces \- a open-r1 Collection \- Hugging Face, [https\://huggingface.co/collections/open-r1/codeforces](https://huggingface.co/collections/open-r1/codeforces)
> 39. [https\://huggingface.co/datasets/open-r1/codeforces](https://huggingface.co/datasets/open-r1/codeforces)
> 40. README.md · open-r1/codeforces-cots at, [https\://huggingface.co/datasets/open-r1/codeforces-cots/blob/177480d489039128dca71d312849c0526482df60/README.md](https://huggingface.co/datasets/open-r1/codeforces-cots/blob/177480d489039128dca71d312849c0526482df60/README.md)
> 41. open-r1/codeforces-cots at main \- Hugging Face, [https\://huggingface.co/datasets/open-r1/codeforces-cots/tree/main](https://huggingface.co/datasets/open-r1/codeforces-cots/tree/main)
> 42. open-r1/codeforces-cots · Datasets at Hugging Face, [https\://huggingface.co/datasets/open-r1/codeforces-cots/viewer/solutions\_decontaminated](https://huggingface.co/datasets/open-r1/codeforces-cots/viewer/solutions_decontaminated)
> 43. open-r1/codeforces-submissions at main \- Hugging Face, [https\://huggingface.co/datasets/open-r1/codeforces-submissions/tree/main](https://huggingface.co/datasets/open-r1/codeforces-submissions/tree/main)
> 44. README.md · open-r1/codeforces-submissions at ... \- Hugging Face, [https\://huggingface.co/datasets/open-r1/codeforces-submissions/blob/07e7bf31f284b0e90e02dfc7ab675d5821e1c068/README.md](https://huggingface.co/datasets/open-r1/codeforces-submissions/blob/07e7bf31f284b0e90e02dfc7ab675d5821e1c068/README.md)
> 45. nvidia/Nemotron-Competitive-Programming-v1 \- Hugging Face, [https\://huggingface.co/datasets/nvidia/Nemotron-Competitive-Programming-v1](https://huggingface.co/datasets/nvidia/Nemotron-Competitive-Programming-v1)
> 46. 4E-AI/Fidel-Competitive-coding · Datasets at Hugging Face, [https\://huggingface.co/datasets/4E-AI/Fidel-Competitive-coding/viewer/](https://huggingface.co/datasets/4E-AI/Fidel-Competitive-coding/viewer/)
> 47. Datasets \- Hugging Face, [https\://huggingface.co/datasets?other=arxiv:2507.17797](https://huggingface.co/datasets?other=arxiv:2507.17797)
> 48. 4E-AI/Fidel-Competitive-coding · Datasets at Hugging Face, [https\://huggingface.co/datasets/4E-AI/Fidel-Competitive-coding/viewer/default/train?p=1](https://huggingface.co/datasets/4E-AI/Fidel-Competitive-coding/viewer/default/train?p=1)
> 49. 4E-AI/Fidel-Competitive-coding · Datasets at Hugging Face, [https\://huggingface.co/datasets/4E-AI/Fidel-Competitive-coding](https://huggingface.co/datasets/4E-AI/Fidel-Competitive-coding)
> 50. 4E-AI/Fidel-Competitive-coding · Discussions \- Hugging Face, [https\://huggingface.co/datasets/4E-AI/Fidel-Competitive-coding/discussions](https://huggingface.co/datasets/4E-AI/Fidel-Competitive-coding/discussions)
> 51. PrimeIntellect/verifiable-coding-problems · Datasets at Hugging Face, [https\://huggingface.co/datasets/PrimeIntellect/verifiable-coding-problems](https://huggingface.co/datasets/PrimeIntellect/verifiable-coding-problems)
> 52. PrimeIntellect/verifiable-coding-problems at refs/convert/parquet, [https\://huggingface.co/datasets/PrimeIntellect/verifiable-coding-problems/blame/refs%2Fconvert%2Fparquet/.gitattributes](https://huggingface.co/datasets/PrimeIntellect/verifiable-coding-problems/blame/refs%2Fconvert%2Fparquet/.gitattributes)
> 53. huggingface/open-r1: Fully open reproduction of DeepSeek-R1, [https\://github.com/huggingface/open-r1](https://github.com/huggingface/open-r1)
> 54. A Survey on Large Language Models for Code Generation \- arXiv, [https\://arxiv.org/html/2406.00515v1](https://arxiv.org/html/2406.00515v1)
> 55. Dataset Selection and Filtering \| Chi-Shan0707/TinyLoRA-GRPO, [https\://deepwiki.com/Chi-Shan0707/TinyLoRA-GRPO-Coder/4.2-dataset-selection-and-filtering](https://deepwiki.com/Chi-Shan0707/TinyLoRA-GRPO-Coder/4.2-dataset-selection-and-filtering)
> 56. A Survey on Large Language Models for Code Generation, [https\://www\.researchgate.net/publication/381126726\_A\_Survey\_on\_Large\_Language\_Models\_for\_Code\_Generation](https://www.researchgate.net/publication/381126726_A_Survey_on_Large_Language_Models_for_Code_Generation)
> 57. Track: San Diego Poster Session 6 \- NeurIPS 2026, [https\://neurips.cc/virtual/2025/loc/san-diego/session/128336](https://neurips.cc/virtual/2025/loc/san-diego/session/128336)
> 58. Methods, Evaluation, and Deployment Trade-Offs \- MDPI, [https\://www\.mdpi.com/2079-9292/15/20/4826/pdf](https://www.mdpi.com/2079-9292/15/20/4826/pdf)
> 59. Efficient Reasoning Models: A Survey \- Hyun\_Jae \- 티스토리, [https\://hyeon-jae.tistory.com/497](https://hyeon-jae.tistory.com/497)
> 60. Efficient Inference for Large Reasoning Models: A ... \- Hugging Face, [https\://huggingface.co/buckets/nielsr/arxiv-chandra-ocr-full-markdown-20260406/tree/2503/2503.23077.md](https://huggingface.co/buckets/nielsr/arxiv-chandra-ocr-full-markdown-20260406/tree/2503/2503.23077.md)
> 61. ConCISE: Confidence-guided Compression in Step-by ... \- SciSpace, [https\://scispace.com/pdf/concise-confidence-guided-compression-in-step-by-step-gba4wsykrjx4.pdf](https://scispace.com/pdf/concise-confidence-guided-compression-in-step-by-step-gba4wsykrjx4.pdf)
> 62. Daily Papers \- Hugging Face, [https\://huggingface.co/papers?q=long-reasoning](https://huggingface.co/papers?q=long-reasoning)
> 63. Generating Structured Outputs from Language Models: Benchmark, [https\://www\.researchgate.net/publication/388231978\_Generating\_Structured\_Outputs\_from\_Language\_Models\_Benchmark\_and\_Studies](https://www.researchgate.net/publication/388231978_Generating_Structured_Outputs_from_Language_Models_Benchmark_and_Studies)
> 64. JSON Schema vs Grammar-Constrained Decoding For Structured, [https\://glitchdata.com/articles/json-schema-vs-grammar-constrained-decoding-for-structured-output](https://glitchdata.com/articles/json-schema-vs-grammar-constrained-decoding-for-structured-output)
> 65. Guaranteeing Valid JSON Output from LLMs \| VidyalAI, [https\://vidyal.ai/blog/guaranteed-valid-json-output/](https://vidyal.ai/blog/guaranteed-valid-json-output/)
> 66. Grammar-Constrained Decoding Makes Large Language Models, [https\://aclanthology.org/2025.acl-industry.34.pdf](https://aclanthology.org/2025.acl-industry.34.pdf)
> 67. Beyond training: engineering vision-language models for immediate, [https\://www\.spiedigitallibrary.org/conference-proceedings-of-spie/14027/140270C/Beyond-training--engineering-vision-language-models-for-immediate-disaster/10.1117/12.3100583.full](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/14027/140270C/Beyond-training--engineering-vision-language-models-for-immediate-disaster/10.1117/12.3100583.full)
> 68. The schema is the contract: structured output from protobuf to agents, [https\://tanchao.xyz/posts/2026/07/31/the-schema-is-the-contract/](https://tanchao.xyz/posts/2026/07/31/the-schema-is-the-contract/)
> 69. LLM Tool Calling Determinism: Production Patterns That Work (2026, [https\://iotdigitaltwinplm.com/llm-tool-calling-determinism-patterns-2026/](https://iotdigitaltwinplm.com/llm-tool-calling-determinism-patterns-2026/)
> 70. Sandlock: Confining AI Agent Code with Unprivileged Linux Primitives, [https\://arxiv.org/html/2605.26298v1](https://arxiv.org/html/2605.26298v1)
> 71. Sandboxing Desktop Linux: Firejail, Bubblewrap, and Flatpak \- Haven, [https\://havenmessenger.com/blog/posts/linux-sandboxing-tools/](https://havenmessenger.com/blog/posts/linux-sandboxing-tools/)
> 72. Linux Application Sandboxing with Bubblewrap, Firejail and Flatpak, [https\://blog.basilisk.com.br/linux-app-sandboxing-bubblewrap-firejail-flatpak](https://blog.basilisk.com.br/linux-app-sandboxing-bubblewrap-firejail-flatpak)
> 73. Use bubblewrap for sandboxing instead of firejail · Issue \#62 \- GitHub, [https\://github.com/orjail/orjail/issues/62](https://github.com/orjail/orjail/issues/62)
> 74. Overview of GGUF quantization methods : r/LocalLLaMA \- Reddit, [https\://www\.reddit.com/r/LocalLLaMA/comments/1ba55rj/overview\_of\_gguf\_quantization\_methods/](https://www.reddit.com/r/LocalLLaMA/comments/1ba55rj/overview_of_gguf_quantization_methods/)
> 75. Best Quantization Methods for llama.cpp \- Sesame Disk, [https\://sesamedisk.com/best-quantization-methods-for-llama-cpp/](https://sesamedisk.com/best-quantization-methods-for-llama-cpp/)
> 76. Quantization Formats Compared cheat sheet, [https\://aiengineeringfromscratch.com/cheatsheets/quantization](https://aiengineeringfromscratch.com/cheatsheets/quantization)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAVCAYAAABc6S4mAAABP0lEQVR4Xu2ULUsEURSGjwgLRosImlRw2Spq2aJYjIJo2GARXJtFlk0W7YKgCBaLH2hVwaBxgyDIKop/YINFMajBj+dyLsOds4wszG7bB54w73vhztw5MyJt2rSSLtzHM3zE4Xidni18wHX8xol4nQ539++4iRnsidfpmcRfnLFFWqaxii+iG9z766lwUTM4xpoNm8kTntsQhvBE9OmucBH78dBn17gUrU7AvWA3NRu28ORF++4gGxHdoKFhGBddPGsLTxnvTLYiOtINURTdYNAWngvcMdkpbpsskV18xQ5bQCe+YQWPAj9wLlj3Lzd4aUPPmOjT9QVZ1me9QZaI+2o/sWQLz6rohIUs47PJ6pgX/S2M4g8OxOsIN7r2rN3575msDvdROddEZ9qSE82/RI+wIPp3PRA9/1tciFYH/AEj6T9BWiDVeAAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAVCAYAAACpF6WWAAABHklEQVR4Xu2UvUpDQRBGR4SAnSDYClr4BGplYRBFsFAQfAVfIAQrG21FCBosUqQKNnaKWNgKio0azBtY2ARFsPHnTGaR3Un04rXNgVPsfMve3bs/In369GII63iCDziZxvmoYBO38R3n0vjv6CxfcA8LOJrG+SjiJ676IA9LeIdPYoPeh/Z83CkvR/joi/+lhae+CBPYEFvFBVbT+Gd0k3S3d3wQmBXLh33wGzNiM1nzQWATb3wxiw2xQXWpvTjDXV/M4hDbOOADGMRnXPFBFtd47ouBafzAkaimH1+M2l3o7XnDsg8CJbx1tWVccLUO62JXckpsJuNp/I0+LgdRewyvxH5LF3rQ1S2xc+jRF0pfLF3FJdbwGF9xP+rX4QtCtzJY3+nc0gAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,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>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAUCAYAAAC07qxWAAAAeklEQVR4XmNgGAWDH2gC8XsgloLy64D4IUIaAUqA+AsQ2wAxCxBnAnECsgIYcAbif0D8H4g/A/FEBogGrMAJiPuA+BoDREMCiiwQrAHie0h8dgaIM5KRxMDgHRB3IvH9GCAe4UcSA4NgID4AxOuAeDsQLwNieWQFAwgAQzIUqsEVqfsAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,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>
