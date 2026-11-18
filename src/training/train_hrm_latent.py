"""Training and fine-tuning pipeline for Sapient-style HRM Latent Reasoner.

Trains:
1. Module H and Module L recurrent weights on high-star (10k+ stars) code invariants
   and competitive programming blueprints.
2. ACT Halting Q-head to converge dynamically without redundant forward recurrence.
3. Invariant projection head to align latent attractor states with ground-truth invariants.
"""

import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from rich.console import Console
from rich.panel import Panel

from hrm_agent.hrm_latent_engine import HRMLatentConfig, HRMLatentReasoner


def load_training_corpus(
    production_code_path: str = "data/curated/production_code_10k_stars.jsonl",
    blueprints_path: str = "data/curated/node1_blueprints.jsonl",
) -> list[dict[str, Any]]:
    """Load combined dataset of 10k+ stars production code invariants and CP blueprints."""
    samples: list[dict[str, Any]] = []

    # 1. Load 10k+ stars production code
    if os.path.exists(production_code_path):
        with open(production_code_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    samples.append(
                        {
                            "input_text": f"{item.get('task', '')} ({item.get('language', '')})",
                            "target_invariant": item.get("invariant", ""),
                        }
                    )

    # 2. Load CP blueprints
    if os.path.exists(blueprints_path):
        with open(blueprints_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    prompt = item.get("prompt", "")
                    completion = item.get("completion", "")
                    samples.append(
                        {
                            "input_text": prompt[:256],
                            "target_invariant": completion[:256],
                        }
                    )

    return samples


def train_hrm_latent(
    output_dir: str = "outputs/checkpoints_hrm_latent",
    epochs: int = 5,
    learning_rate: float = 3e-4,
    device: str = "auto",
) -> str:
    """Train HRM latent recurrent reasoner on invariant discovery corpus."""
    console = Console()
    console.print(Panel.fit("[bold magenta]Training HRM Latent Recurrent Reasoner[/bold magenta]"))

    dev = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
    console.print(f"[cyan]Compute Device:[/cyan] {dev}")

    corpus = load_training_corpus()
    if not corpus:
        raise ValueError("Training corpus is empty. Ensure data/curated/ has valid jsonl files.")

    console.print(f"[cyan]Loaded training samples:[/cyan] {len(corpus)}")

    config = HRMLatentConfig(
        hidden_size=256,
        num_heads=4,
        H_cycles=2,
        L_cycles=3,
        H_layers=2,
        L_layers=2,
        vocab_size=32000,
    )
    model = HRMLatentReasoner(config).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    model.train()
    target_embedder = nn.Embedding(config.vocab_size, config.hidden_size).to(dev)

    for epoch in range(epochs):
        total_loss = 0.0
        for sample in corpus:
            inp_str = sample["input_text"]
            tgt_str = sample["target_invariant"]

            inp_tokens = torch.tensor([[min(ord(c), 31999) for c in inp_str[:128]]], device=dev)
            tgt_tokens = torch.tensor([[min(ord(c), 31999) for c in tgt_str[:128]]], device=dev)
            if inp_tokens.shape[1] == 0:
                inp_tokens = torch.tensor([[1, 2, 3]], device=dev)
            if tgt_tokens.shape[1] == 0:
                tgt_tokens = torch.tensor([[4, 5, 6]], device=dev)

            optimizer.zero_grad()

            embeds = model.embed_tokens(inp_tokens)
            carry = model.init_carry(embeds.shape[0], embeds.shape[1])

            # Forward recurrent step
            carry, concept_repr, halt_logits = model.forward_recurrent_step(embeds, carry)

            # Target embedding representation
            with torch.no_grad():
                tgt_repr = target_embedder(tgt_tokens).mean(dim=1)

            # Latent alignment loss (cosine similarity distance)
            pred_pooled = concept_repr.mean(dim=1)
            align_loss = 1.0 - torch.cosine_similarity(pred_pooled, tgt_repr).mean()

            # Halting loss (reinforcing convergence around step 3-4)
            halt_target = torch.tensor([1 if carry.step >= 3 else 0], device=dev)
            halt_loss = nn.functional.cross_entropy(halt_logits, halt_target)

            loss = align_loss + 0.1 * halt_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(corpus)
        console.print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_file = out_path / "hrm_latent_reasoner.pt"
    torch.save({"config": config.model_dump(), "state_dict": model.state_dict()}, model_file)
    console.print(f"[bold green]Saved trained HRM Latent model to {model_file}[/bold green]")
    return str(model_file)


if __name__ == "__main__":
    train_hrm_latent()
