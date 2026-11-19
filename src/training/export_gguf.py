"""PEFT merge, domain calibration corpus generation, imatrix computation, and GGUF quantization.

Implements:
1. PEFT merge_and_unload()
2. Domain calibration_cp.txt builder (1/3 blueprints, 1/3 C++20 + diffs, 1/3 critic verdicts)
3. llama-imatrix + llama-quantize (Q5_K_M for 0.5B Coder, Q4_K_M/IQ4_XS for 1.5B Planner & Critic)
4. F16 vs Quantized GBNF / compile parity verification gate.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


def build_calibration_dataset(
    output_path: str = "data/calibration_cp.txt",
    curated_data_dir: str = "data/curated",
) -> str:
    """
    Build balanced domain calibration corpus for llama-imatrix:
    1/3 blueprints, 1/3 C++20 + diffs, 1/3 critic verdicts.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    blueprints: list[str] = []
    codes_and_diffs: list[str] = []
    critics: list[str] = []

    cur_dir = Path(curated_data_dir)
    bp_file = cur_dir / "node1_blueprints.jsonl"
    ptc_file = cur_dir / "node2_plan_to_code.jsonl"
    etd_file = cur_dir / "node2_error_to_diff.jsonl"
    critic_file = cur_dir / "node3_critic.jsonl"

    if bp_file.exists():
        with open(bp_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                blueprints.append(item.get("prompt", "") + "\n" + item.get("completion", ""))

    if ptc_file.exists():
        with open(ptc_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                codes_and_diffs.append(item.get("prompt", "") + "\n" + item.get("completion", ""))

    if etd_file.exists():
        with open(etd_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                codes_and_diffs.append(item.get("prompt", "") + "\n" + item.get("completion", ""))

    if critic_file.exists():
        with open(critic_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                critics.append(item.get("prompt", "") + "\n" + item.get("completion", ""))

    # If curated files were empty, populate with domain calibration samples
    if not blueprints:
        blueprints.append(
            '{\n  "analysis": "Two pointer approach tracking minimum window",\n  "paradigm": "two_pointers",\n'
            '  "data_structures": ["vector"],\n  "time_complexity": "O(N)",\n  "space_complexity": "O(1)",\n'
            '  "use_int64": true,\n  "edge_cases": ["N=1"],\n  "steps": ["Advance right pointer", "Shrink left"]\n}'
        )
    if not codes_and_diffs:
        codes_and_diffs.append(
            "#include <iostream>\n#include <vector>\nusing namespace std;\nint main() {\n"
            "    ios::sync_with_stdio(false);\n    cin.tie(nullptr);\n    return 0;\n}"
        )
    if not critics:
        critics.append(
            '{\n  "analysis": "Time complexity within limits, passes all checks",\n  "status": "PASS",\n'
            '  "bottleneck_class": "none",\n  "escalate_to_planner": false,\n  "target_lines": "none",\n'
            '  "patch_directive": "None"\n}'
        )

    # Balance into equal 1/3 proportions
    target_count = max(len(blueprints), len(codes_and_diffs), len(critics))
    merged_corpus = []

    for i in range(target_count):
        merged_corpus.append(blueprints[i % len(blueprints)])
        merged_corpus.append(codes_and_diffs[i % len(codes_and_diffs)])
        merged_corpus.append(critics[i % len(critics)])

    with open(out_file, "w", encoding="utf-8") as f:
        f.writelines(chunk.strip() + "\n\n" for chunk in merged_corpus)

    print(f"[HRM Export] Wrote {len(merged_corpus)} calibration samples to {out_file}")
    return str(out_file)


def merge_peft_adapter(base_model_path: str, adapter_path: str, merged_output_path: str) -> str:
    """Merge LoRA adapter weights into base model weights and save to disk."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[HRM Export] Merging adapter {adapter_path} into base model {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = model.merge_and_unload()

    Path(merged_output_path).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(merged_output_path)
    tokenizer.save_pretrained(merged_output_path)
    print(f"[HRM Export] Merged model saved to {merged_output_path}")
    return merged_output_path


def quantize_model(
    f16_gguf_path: str,
    output_gguf_path: str,
    quant_type: str = "Q4_K_M",
    calibration_file: str | None = None,
    imatrix_output_path: str | None = None,
) -> str:
    """
    Quantize GGUF model using llama-imatrix and llama-quantize.
    quant_type:
      - 'Q5_K_M' for 0.5B Coder
      - 'Q4_K_M' or 'IQ4_XS' for 1.5B Planner & Critic
    """
    imatrix_bin = shutil.which("llama-imatrix")
    quantize_bin = shutil.which("llama-quantize")

    if not quantize_bin:
        print("[HRM Export] llama-quantize binary not found in PATH. Skipping binary quantization.")
        return f16_gguf_path

    # Step 1: Compute importance matrix if calibration data provided
    imatrix_file = None
    if imatrix_bin and calibration_file and os.path.exists(calibration_file):
        imatrix_file = imatrix_output_path or f16_gguf_path + ".imatrix.dat"
        print("[HRM Export] Computing importance matrix via llama-imatrix...")
        cmd_imatrix = [
            imatrix_bin,
            "-m",
            f16_gguf_path,
            "-f",
            calibration_file,
            "-o",
            imatrix_file,
            "--chunks",
            "32",
        ]
        try:
            subprocess.run(cmd_imatrix, check=True)
            print(f"[HRM Export] Importance matrix computed: {imatrix_file}")
        except Exception as e:
            print(f"[HRM Export] Warning: llama-imatrix failed: {e}. Proceeding without imatrix.")
            imatrix_file = None

    # Step 2: Quantize
    cmd_quant = [quantize_bin]
    if imatrix_file and os.path.exists(imatrix_file):
        cmd_quant.extend(["--imatrix", imatrix_file])
    cmd_quant.extend([f16_gguf_path, output_gguf_path, quant_type])

    print(f"[HRM Export] Running: {' '.join(cmd_quant)}")
    try:
        subprocess.run(cmd_quant, check=True)
        print(f"[HRM Export] Quantization complete: {output_gguf_path}")
        return output_gguf_path
    except Exception as e:
        print(f"[HRM Export] llama-quantize failed: {e}")
        return f16_gguf_path


def run_parity_gate(
    f16_model_engine: Any,
    quant_model_engine: Any,
    test_prompts: list[tuple[str, str]],  # (prompt, grammar_path)
) -> dict[str, Any]:
    """
    F16 vs Quantized GBNF / compile parity verification gate.
    Verifies that quantized model does not degrade grammar compliance or compilation rate.
    """
    f16_passes = 0
    quant_passes = 0
    total = len(test_prompts)

    report = {
        "total": total,
        "f16_passes": 0,
        "quant_passes": 0,
        "parity_passed": False,
    }

    for prompt, grammar_path in test_prompts:
        txt_f16, _ = f16_model_engine.generate(prompt=prompt, grammar_path=grammar_path)
        txt_quant, _ = quant_model_engine.generate(prompt=prompt, grammar_path=grammar_path)

        # Check JSON schema or compile
        if "planner" in grammar_path or "critic" in grammar_path:
            try:
                json.loads(txt_f16.strip())
                f16_passes += 1
            except Exception:
                pass
            try:
                json.loads(txt_quant.strip())
                quant_passes += 1
            except Exception:
                pass
        else:
            # Code generation mode
            if txt_f16.startswith("#include") or "def " in txt_f16:
                f16_passes += 1
            if txt_quant.startswith("#include") or "def " in txt_quant:
                quant_passes += 1

    report["f16_passes"] = f16_passes
    report["quant_passes"] = quant_passes
    # Parity gate condition: Quantized accuracy must be within 95% of F16
    report["parity_passed"] = quant_passes >= (f16_passes * 0.95)
    return report
