"""RTX 3080 (10 GB VRAM) bf16 LoRA SFT and DPO Training Pipeline.

Configured with:
- r=32, lora_alpha=64, all 7 linear projection layers
- max_seq_length=2048
- adamw_8bit optimizer
- Liger-kernel chunked cross-entropy (fits <6.5 GB on a 10 GB VRAM GPU)
- Gradient checkpointing
- TRL DPOTrainer with PEFT adapter-disabled reference model (no duplicate base model weights)
"""

import os
from typing import Any

import yaml

try:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        TrainingArguments,
    )
except ImportError:
    torch = None


def apply_liger_cross_entropy():
    """Apply Liger chunked cross-entropy kernel to Qwen models for memory efficiency."""
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2

        apply_liger_kernel_to_qwen2()
        print("[HRM Training] Applied Liger chunked cross-entropy kernel for Qwen2.")
    except Exception as e:
        print(f"[HRM Training] Liger kernel not activated: {e}. Falling back to standard cross-entropy.")


def get_lora_config(r: int = 32, lora_alpha: int = 64, lora_dropout: float = 0.05) -> Any:
    """LoRA configuration targeting all 7 linear projection layers."""
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def train_sft(
    base_model_name_or_path: str,
    train_data_path: str,
    output_dir: str = "outputs/checkpoints",
    config_path: str = "configs/training.yaml",
    epochs: int | None = None,
    learning_rate: float | None = None,
    batch_size: int | None = None,
    grad_accum: int | None = None,
) -> str:
    """Run bf16 LoRA SFT training within the 10GB RTX 3080 / 6GB RTX 4050 VRAM budget."""
    if torch is None:
        raise RuntimeError("PyTorch and Transformers are required for training.")

    # Load YAML hyperparameters if available
    train_cfg: dict[str, Any] = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            train_cfg = cfg.get("training", {})

    max_seq_len = train_cfg.get("max_seq_length", 2048)
    lr = learning_rate if learning_rate is not None else train_cfg.get("learning_rate", 2e-4)
    bs = batch_size if batch_size is not None else train_cfg.get("per_device_train_batch_size", 1)
    ga = grad_accum if grad_accum is not None else train_cfg.get("gradient_accumulation_steps", 4)
    ep = epochs if epochs is not None else train_cfg.get("num_train_epochs", 3)
    use_bf16 = train_cfg.get("bf16", True) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    if train_cfg.get("use_liger_kernel", True):
        apply_liger_cross_entropy()

    print(f"[HRM SFT] Loading base model: {base_model_name_or_path}")
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # Enable gradient checkpointing to keep VRAM usage < 4.5 GB
    model.gradient_checkpointing_enable()

    # Wrap model with LoRA
    lora_config = get_lora_config(r=32, lora_alpha=64)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    raw_dataset = load_dataset("json", data_files=train_data_path, split="train")

    def tokenize_fn(examples):
        prompts = examples["prompt"]
        completions = examples["completion"]
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        min_completion_len = 32
        max_prompt_len = max(64, max_seq_len - min_completion_len)

        for p, c in zip(prompts, completions):
            # Reserve capacity for completion to guarantee supervised tokens
            prompt_enc = tokenizer(p + "\n", truncation=True, max_length=max_prompt_len, padding=False)
            prompt_ids = prompt_enc["input_ids"]
            comp_enc = tokenizer(c, truncation=True, max_length=max_seq_len - len(prompt_ids), padding=False)
            comp_ids = comp_enc["input_ids"]

            # If completion is completely empty, provide at least the eos token
            if not comp_ids:
                comp_ids = [tokenizer.eos_token_id or 0]

            input_ids = prompt_ids + comp_ids
            attn_mask = [1] * len(input_ids)
            labels = [-100] * len(prompt_ids) + list(comp_ids)

            # Ensure truncation to max_seq_len and at least one supervised completion token
            if len(input_ids) > max_seq_len:
                input_ids = input_ids[:max_seq_len]
                attn_mask = attn_mask[:max_seq_len]
                labels = labels[:max_seq_len]

            # Guarantee at least one valid label
            if not any(lbl != -100 for lbl in labels):
                labels[-1] = input_ids[-1]

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attn_mask)

        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
            "attention_mask": attention_mask_list,
        }

    tokenized_dataset = raw_dataset.map(tokenize_fn, batched=True, remove_columns=raw_dataset.column_names)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=ga,
        num_train_epochs=ep,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=1,
        weight_decay=0.01,
        bf16=use_bf16,
        fp16=False,
        optim="adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
    )

    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt"),
    )

    print("[HRM SFT] Commencing training...")
    trainer.train()

    final_adapter_path = os.path.join(output_dir, "final_adapter")
    model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)
    print(f"[HRM SFT] Successfully saved LoRA adapter to {final_adapter_path}")
    return final_adapter_path


def train_dpo(
    base_model_name_or_path: str,
    adapter_path: str,
    dpo_data_path: str,
    output_dir: str = "outputs/checkpoints_dpo",
    epochs: int | None = None,
    learning_rate: float | None = None,
    grad_accum: int | None = None,
) -> str:
    """Run DPO training using PEFT adapter-disabled reference model (no duplicate weights in VRAM)."""
    try:
        from trl import DPOConfig, DPOTrainer
    except ImportError:
        raise RuntimeError("TRL is required for DPO training. Install via `uv pip install trl`.")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)

    dpo_dataset = load_dataset("json", data_files=dpo_data_path, split="train")

    ep = epochs if epochs is not None else 2
    lr = learning_rate if learning_rate is not None else 5e-5
    ga = grad_accum if grad_accum is not None else 4

    dpo_config = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=ga,
        num_train_epochs=ep,
        learning_rate=lr,
        bf16=use_bf16,
        optim="adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="epoch",
        max_length=2048,
    )

    import inspect

    dpo_kwargs = {
        "model": model,
        "ref_model": None,
        "args": dpo_config,
        "train_dataset": dpo_dataset,
    }
    sig = inspect.signature(DPOTrainer.__init__)
    if "processing_class" in sig.parameters:
        dpo_kwargs["processing_class"] = tokenizer
    else:
        dpo_kwargs["tokenizer"] = tokenizer

    # When ref_model=None, DPOTrainer disables adapter during ref forward pass
    # saving ~3-4 GB VRAM and fitting easily in 6-10 GB VRAM!
    trainer = DPOTrainer(**dpo_kwargs)

    print("[HRM DPO] Starting adapter-disabled reference DPO training...")
    trainer.train()

    final_dpo_path = os.path.join(output_dir, "final_dpo_adapter")
    model.save_pretrained(final_dpo_path)
    tokenizer.save_pretrained(final_dpo_path)
    print(f"[HRM DPO] Saved DPO adapter to {final_dpo_path}")
    return final_dpo_path
