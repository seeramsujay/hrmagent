"""Command Line Interface for HRM Competitive Programming Agent.

Provides commands:
- mine-data: Curate, filter, and balance competitive programming training data
- train-lora: Train 10GB VRAM bf16 LoRA with adamw_8bit and chunked CE
- export-gguf: Merge PEFT, build domain calibration, compute imatrix, and quantize GGUF
- solve: Run the three-node HRM closed-loop pipeline on a single problem
- eval-benchmark: Run full 3-way ablation and stratified benchmark evaluation
"""

import json
import os
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from hrm_agent.hrm_loop import HRMAgentLoop
from hrm_agent.llm_engine import LlamaCppEngine, MockLLMEngine
from hrm_agent.schemas import ProblemSpec
from judge.benchmark_eval import BenchmarkEvaluator
from judge.sandbox import SandboxRunner
from training.dataset_miner import DatasetMiner
from training.export_gguf import (
    build_calibration_dataset,
    merge_peft_adapter,
    quantize_model,
)

app = typer.Typer(
    name="hrm",
    help="Production-Grade Local-First HRM Competitive Programming Agent CLI",
    add_completion=False,
)
console = Console()


@app.command("mine-data")
def mine_data(
    output_dir: str = typer.Option("data/curated", help="Destination path for curated JSONL datasets"),
    min_rating: int = typer.Option(800, help="Minimum rating bound"),
    max_rating: int = typer.Option(1900, help="Maximum rating bound"),
    gm_cutoff: int = typer.Option(2200, help="Upper Grandmaster cutoff"),
    seed_synthetic: bool = typer.Option(True, help="Seed synthetic problems and trajectories"),
    include_codechef: bool = typer.Option(True, help="Include CodeChef competitive programming challenges"),
    include_game_theory: bool = typer.Option(True, help="Include Game Theory competitive programming challenges"),
    codechef_json: Optional[str] = typer.Option(
        "data/codechef_raw.json", help="Path to local or cached CodeChef JSON file"
    ),
):
    """Run Hugging Face + DuckDB curation and trajectory mining pipeline."""
    console.print(Panel.fit("[bold blue]Starting HRM Dataset Mining Pipeline[/bold blue]"))
    miner = DatasetMiner(
        output_dir=output_dir,
        min_rating=min_rating,
        max_rating=max_rating,
        gm_cutoff=gm_cutoff,
    )

    if seed_synthetic:
        console.print("[green]Seeding offline synthetic problems and trajectory pairs...[/green]")
        miner.populate_synthetic_seed()

    if include_codechef:
        console.print("[green]Mining CodeChef challenges into DuckDB...[/green]")
        cc_count = miner.populate_codechef_challenges(json_path=codechef_json)
        console.print(f"[bold cyan]Successfully integrated {cc_count} CodeChef challenges![/bold cyan]")

    if include_game_theory:
        console.print("[green]Mining Game Theory challenges (Sprague-Grundy, Nim, Subtraction games)...[/green]")
        gt_count = miner.populate_game_theory_challenges()
        console.print(f"[bold cyan]Successfully integrated {gt_count} Game Theory challenges![/bold cyan]")

    paths = miner.build_balanced_datasets()
    table = Table(title="Generated Balanced Datasets")
    table.add_column("Dataset Component", style="cyan")
    table.add_column("File Path", style="green")

    for name, path in paths.items():
        table.add_row(name, path)
    console.print(table)
    console.print("[bold green]Dataset mining successfully completed![/bold green]")


@app.command("train-lora")
def train_lora(
    config_path: str = typer.Option("configs/training.yaml", help="Path to training config"),
    mode: str = typer.Option("sft", help="Training mode: 'sft' or 'dpo'"),
    train_data: Optional[str] = typer.Option(None, help="Path to training jsonl"),
    base_model: Optional[str] = typer.Option(None, help="Base model identifier or path"),
    output_dir: str = typer.Option("outputs/checkpoints", help="Output directory"),
    adapter_path: Optional[str] = typer.Option(None, help="Input adapter path for DPO training"),
    epochs: Optional[int] = typer.Option(None, help="Number of training epochs"),
    grad_accum: Optional[int] = typer.Option(None, help="Gradient accumulation steps"),
    learning_rate: Optional[float] = typer.Option(None, help="Learning rate override"),
):
    """Train bf16 LoRA adapter on single GPU with adamw_8bit and chunked CE."""
    console.print(Panel.fit(f"[bold magenta]HRM LoRA Training — Mode: {mode.upper()}[/bold magenta]"))
    from training.train_sft_lora import train_dpo, train_sft

    # Load config defaults
    chosen_base = base_model
    if chosen_base is None and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            model_cfg = cfg.get("model", {})
            data_str = (train_data or "").lower()
            if "node2" in data_str:
                chosen_base = model_cfg.get("coder_base", "Qwen/Qwen2.5-Coder-0.5B-Instruct")
            else:
                chosen_base = model_cfg.get("coder_base", "Qwen/Qwen2.5-Coder-0.5B-Instruct")
    if chosen_base is None:
        chosen_base = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

    if mode == "sft":
        data_file = train_data or "data/curated/node2_plan_to_code.jsonl"
        saved_adapter = train_sft(
            base_model_name_or_path=chosen_base,
            train_data_path=data_file,
            output_dir=output_dir,
            config_path=config_path,
            epochs=epochs,
            learning_rate=learning_rate,
            grad_accum=grad_accum,
        )
        console.print(f"[bold green]LoRA SFT training complete. Adapter at {saved_adapter}[/bold green]")
    elif mode == "dpo":
        dpo_data = train_data or "data/curated/node2_dpo.jsonl"
        # Resolve input SFT adapter path
        resolved_adapter = adapter_path
        if not resolved_adapter:
            candidate_paths = [
                os.path.join(output_dir, "final_adapter"),
                "outputs/checkpoints_plan_to_code/final_adapter",
                "outputs/checkpoints/final_adapter",
                "outputs/checkpoints_architect_critic/final_adapter",
            ]
            for cp in candidate_paths:
                if os.path.exists(cp):
                    resolved_adapter = cp
                    break
        if not resolved_adapter or not os.path.exists(resolved_adapter):
            raise FileNotFoundError("Cannot find input adapter for DPO training. Specify --adapter-path explicitly.")

        dpo_out = output_dir if output_dir != "outputs/checkpoints" else "outputs/checkpoints_dpo"
        final_dpo = train_dpo(
            base_model_name_or_path=chosen_base,
            adapter_path=resolved_adapter,
            dpo_data_path=dpo_data,
            output_dir=dpo_out,
            epochs=epochs,
            learning_rate=learning_rate,
            grad_accum=grad_accum,
        )
        console.print(f"[bold green]DPO training complete. Adapter at {final_dpo}[/bold green]")


@app.command("export-gguf")
def export_gguf(
    base_model: str = typer.Option("Qwen/Qwen2.5-Coder-1.5B-Instruct", help="Base HF model"),
    adapter_path: str = typer.Option("outputs/checkpoints/final_adapter", help="LoRA adapter dir"),
    output_gguf: str = typer.Option("models/gguf/model_q4_k_m.gguf", help="Target GGUF file"),
    quant_type: str = typer.Option("Q4_K_M", help="Quantization type (Q5_K_M, Q4_K_M, IQ4_XS)"),
    calibration_file: str = typer.Option("data/calibration_cp.txt", help="Calibration corpus path"),
):
    """Merge LoRA, build calibration corpus, compute imatrix, and export quantized GGUF."""
    console.print(Panel.fit("[bold cyan]Exporting and Quantizing Model to GGUF[/bold cyan]"))

    cal_path = build_calibration_dataset(output_path=calibration_file)
    merged_dir = "outputs/merged_model"
    merge_peft_adapter(base_model, adapter_path, merged_dir)

    console.print(f"[green]Quantizing model to {quant_type}...[/green]")
    f16_dummy = f"outputs/{Path(base_model).name}_f16.gguf"
    quant_path = quantize_model(
        f16_gguf_path=f16_dummy,
        output_gguf_path=output_gguf,
        quant_type=quant_type,
        calibration_file=cal_path,
    )
    console.print(f"[bold green]Model exported and saved to {quant_path}[/bold green]")


@app.command("solve")
def solve_problem(
    problem_file: str = typer.Option(..., help="Path to JSON problem specification file"),
    language: str = typer.Option("cpp", help="Target language ('cpp' or 'python')"),
    mock: bool = typer.Option(False, help="Use deterministic mock LLM for testing/demo"),
    trace_output: str | None = typer.Option(None, help="Optional JSONL file to persist trace"),
):
    """Solve a competitive programming problem using the 3-node HRM closed-loop pipeline."""
    console.print(Panel.fit(f"[bold yellow]Solving Problem: {problem_file}[/bold yellow]"))

    with open(problem_file, "r", encoding="utf-8") as f:
        spec_data = json.load(f)
    spec = ProblemSpec.model_validate(spec_data)

    if mock:
        planner_engine = MockLLMEngine()
        coder_engine = MockLLMEngine()
        critic_engine = MockLLMEngine()
    else:
        # Load from configs/models.yaml
        with open("configs/models.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        planner_cfg = cfg["option_a"]["node1_planner"]
        coder_cfg = cfg["option_a"]["node2_coder"]
        critic_cfg = cfg["option_a"]["node3_critic"]

        planner_engine = LlamaCppEngine(
            model_path=planner_cfg["model_path"],
            n_ctx=planner_cfg.get("context_length", 2048),
            type_k=planner_cfg.get("type_k", "q8_0"),
            type_v=planner_cfg.get("type_v", "q8_0"),
        )
        coder_engine = LlamaCppEngine(
            model_path=coder_cfg["model_path"],
            n_ctx=coder_cfg.get("context_length", 2048),
            type_k=coder_cfg.get("type_k", "q8_0"),
            type_v=coder_cfg.get("type_v", "q8_0"),
        )
        critic_engine = LlamaCppEngine(
            model_path=critic_cfg["model_path"],
            n_ctx=critic_cfg.get("context_length", 2048),
            type_k=critic_cfg.get("type_k", "q8_0"),
            type_v=critic_cfg.get("type_v", "q8_0"),
        )

    runner = SandboxRunner()
    loop = HRMAgentLoop(
        node1_planner=planner_engine,
        node2_coder=coder_engine,
        node3_critic=critic_engine,
        sandbox=runner,
        language=language,
        trace_file=trace_output,
    )

    trace = loop.solve(spec)

    table = Table(title=f"HRM Execution Summary — Problem: {spec.id}")
    table.add_column("Parameter", style="cyan")
    table.add_column("Result", style="bold green" if trace.solved else "bold red")

    table.add_row("Status", "SOLVED" if trace.solved else "FAILED")
    table.add_row("Final Verdict", trace.final_verdict)
    table.add_row("Macro Steps (Node 1)", str(trace.macro_steps))
    table.add_row("Micro Iterations (Node 2)", str(trace.micro_iterations))
    table.add_row("Critic Iterations (Node 3)", str(trace.critic_iterations))

    console.print(table)

    if trace.final_code:
        syntax = Syntax(
            trace.final_code,
            "cpp" if language == "cpp" else "python",
            theme="monokai",
            line_numbers=True,
        )
        console.print(Panel(syntax, title="Final Synthesized Code"))


@app.command("eval-benchmark")
def eval_benchmark(
    problems_dir: str = typer.Option("data/benchmark", help="Directory of problem JSON specs"),
    language: str = typer.Option("cpp", help="Target language ('cpp' or 'python')"),
    report_path: str = typer.Option("outputs/eval_report.json", help="Path to save report"),
    mock: bool = typer.Option(True, help="Use mock engine for benchmark demonstration"),
):
    """Run 3-way ablation evaluation suite across rating buckets."""
    console.print(Panel.fit("[bold blue]Running HRM Benchmark & Ablation Suite[/bold blue]"))

    problems = []
    p_dir = Path(problems_dir)
    if p_dir.exists():
        for file in p_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                problems.append(ProblemSpec.model_validate(data))
            except Exception:
                pass

    if not problems:
        console.print("[yellow]No problem files found in dir. Generating synthetic benchmark set...[/yellow]")
        # 5 Synthetic problems spanning rating buckets
        synthetic_specs = [
            ProblemSpec(
                id="SYNTH-101",
                title="Minimal Array Sum",
                description="Given array of N integers, compute total sum.",
                input_format="T test cases. N, followed by N integers.",
                output_format="Sum per test case.",
                constraints="1 <= N <= 1000",
                rating=900,
                sample_inputs=["1\n3\n1 2 3\n"],
                sample_outputs=["6\n"],
            ),
            ProblemSpec(
                id="SYNTH-201",
                title="Two Pointer Window",
                description="Find smallest window containing all distinct values.",
                input_format="N integers.",
                output_format="Length of minimum window.",
                constraints="1 <= N <= 2*10^5",
                rating=1350,
                sample_inputs=["1\n5\n1 2 2 3 1\n"],
                sample_outputs=["4\n"],
            ),
            ProblemSpec(
                id="SYNTH-301",
                title="Tree Diameter",
                description="Find the maximum distance between any two nodes in a tree.",
                input_format="N, followed by N-1 edges.",
                output_format="Integer diameter.",
                constraints="1 <= N <= 2*10^5",
                rating=1650,
                sample_inputs=["1\n3\n1 2\n2 3\n"],
                sample_outputs=["2\n"],
            ),
            ProblemSpec(
                id="SYNTH-401",
                title="Segment Tree Point Updates",
                description="Process Q range sum queries with point updates.",
                input_format="N and Q, array, then Q queries.",
                output_format="Result of type 2 queries.",
                constraints="1 <= N, Q <= 2*10^5",
                rating=1950,
                sample_inputs=["1\n3 1\n1 2 3\n2 1 3\n"],
                sample_outputs=["6\n"],
            ),
            ProblemSpec(
                id="SYNTH-501",
                title="Biconnected Components Overhang",
                description="Count bridges and articulation points.",
                input_format="N, M and edges.",
                output_format="Two space-separated integers.",
                constraints="1 <= N <= 2*10^5",
                rating=2250,
                sample_inputs=["1\n4 3\n1 2\n2 3\n3 4\n"],
                sample_outputs=["3 2\n"],
            ),
        ]
        problems = synthetic_specs

    evaluator = BenchmarkEvaluator(
        node1_planner=MockLLMEngine(),
        node2_coder=MockLLMEngine(),
        node3_critic=MockLLMEngine(),
        language=language,
    )

    results = evaluator.evaluate_ablation(problems, output_report_path=report_path)
    evaluator.print_summary_tables(results)
    console.print(f"[bold green]Benchmark evaluation report written to {report_path}[/bold green]")


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", help="Host address to bind"),
    port: int = typer.Option(8000, help="Port to listen on"),
    mode: str = typer.Option("dual-pair", help="Inference mode ('dual-pair' or 'mock')"),
    device: str = typer.Option("auto", help="Execution device ('auto', 'cuda', 'cpu')"),
    role1_path: str = typer.Option("outputs/merged/role1_architect_critic", help="Path to Role 1 merged model"),
    role2_path: str = typer.Option("outputs/merged/role2_coder_patcher", help="Path to Role 2 merged model"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
    api_key: Optional[str] = typer.Option(None, help="Optional API key for Bearer authentication"),
):
    """Launch the localhost HRM Agent server with OpenAI and closed-loop endpoints."""
    console.print(
        Panel.fit(
            f"[bold cyan]Launching HRM Agent Localhost Server on http://{host}:{port}[/bold cyan]\n"
            f"[dim]Mode: {mode} | Device: {device}[/dim]\n"
            f"[green]OpenAI Endpoints: /v1/chat/completions, /v1/models\n"
            f"HRM Endpoints: /solve, /evaluate, /health[/green]"
        )
    )
    import uvicorn

    if reload:
        uvicorn.run("hrm_agent.server:app", host=host, port=port, reload=True)
    else:
        from hrm_agent.server import create_app

        server_app = create_app(
            mode=mode,
            device=device,
            role1_path=role1_path,
            role2_path=role2_path,
            api_key=api_key,
        )
        uvicorn.run(server_app, host=host, port=port, reload=False)


@app.command("harness")
def test_harness(
    server_url: str = typer.Option("http://127.0.0.1:8000", help="URL of running HRM server"),
    problems_file: str = typer.Option("data/codechef_raw.json", help="Path to benchmark JSON corpus"),
    limit: int = typer.Option(5, help="Number of challenges to benchmark"),
    language: str = typer.Option("python", help="Target language ('python' or 'cpp')"),
    report_path: str = typer.Option("outputs/harness_report.json", help="Output path for harness report"),
):
    """Run automated evaluation harness against the HRM server via HTTP."""
    from hrm_agent.harness import run_benchmark_harness

    run_benchmark_harness(
        server_url=server_url,
        problems_file=problems_file,
        limit=limit,
        language=language,
        report_path=report_path,
    )


@app.command("merge-adapters")
def merge_adapters(
    base_model: str = typer.Option("Qwen/Qwen2.5-Coder-0.5B-Instruct", help="Base Hugging Face model identifier"),
    role1_adapter: str = typer.Option("outputs/checkpoints_architect_critic/final_adapter", help="Role 1 LoRA path"),
    role2_adapter: str = typer.Option("outputs/checkpoints_dpo/final_dpo_adapter", help="Role 2 LoRA path"),
    role1_output: str = typer.Option("outputs/merged/role1_architect_critic", help="Destination for merged Role 1"),
    role2_output: str = typer.Option("outputs/merged/role2_coder_patcher", help="Destination for merged Role 2"),
):
    """Merge trained LoRA adapters into standalone Hugging Face weight directories."""
    console.print(Panel.fit("[bold magenta]Merging HRM Dual-Pair LoRA Adapters[/bold magenta]"))
    console.print(f"[cyan]Role 1 (Architect & Critic): {role1_adapter} -> {role1_output}[/cyan]")
    m1 = merge_peft_adapter(base_model, role1_adapter, role1_output)
    console.print(f"[bold green]Role 1 merged successfully at {m1}[/bold green]")

    console.print(f"[cyan]Role 2 (Fast Coder & Patcher): {role2_adapter} -> {role2_output}[/cyan]")
    m2 = merge_peft_adapter(base_model, role2_adapter, role2_output)
    console.print(f"[bold green]Role 2 merged successfully at {m2}[/bold green]")


@app.command("terminalbench")
def run_terminal_bench(
    report_path: str = typer.Option("outputs/terminal_bench_report.json", help="Path to save benchmark report JSON"),
):
    """Run agentic TerminalBench diagnostics & execution benchmark suite."""
    console.print(Panel.fit("[bold magenta]Running TerminalBench Benchmark Suite[/bold magenta]"))
    from judge.terminal_bench import TerminalBenchEvaluator

    evaluator = TerminalBenchEvaluator()
    summary = evaluator.evaluate_all(output_report_path=report_path)
    console.print(
        f"[bold green]TerminalBench Complete! Pass Rate: {summary['pass_rate_pct']}% "
        f"({summary['passed_tasks']}/{summary['total_tasks']})[/bold green]"
    )


if __name__ == "__main__":
    app()
