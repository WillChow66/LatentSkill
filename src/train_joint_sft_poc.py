"""Plan A POC: End-to-end joint SFT (Composer + 7B model trained together).

This is the proof-of-concept variant of train_composer.py, with the model
unfrozen and added to the optimizer. Validates that the joint gradient flow
works correctly (model + query_latents both update).

Differences from train_composer.py:
  - model is trainable (no .eval(), no requires_grad=False)
  - optimizer includes both query_latents AND model.parameters()
  - separate LR groups: model gets 1e-5, query_latents gets 5e-3
  - encode_skill removes the no_grad context (gradient flows through encoder embed)
  - gradient checkpointing enabled for memory

Single-GPU only. For multi-GPU FSDP, see train_joint_sft_fsdp.py (TODO).

Usage:
  python -m src.train_joint_sft_poc \
      --model-path /work/.../latent_skills_token_v2 \
      --sft-data /path/to/sft.parquet \
      --skills-json SkillRL/memory_data/alfworld/claude_style_skills.json \
      --output-dir /work/.../joint_sft_poc \
      --epochs 1 \
      --max-samples 50 \
      --model-lr 1e-5 \
      --latent-lr 5e-3
"""

import json
import hashlib
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.train_composer import (
    SKILLS_START_MARKER,
    SKILLS_END_MARKER,
    hash_skill,
    extract_individual_skills,
    parse_skills_from_content,
    ComposerDistillationDataset,
)

logger = logging.getLogger(__name__)


class JointComposer(nn.Module):
    """Composer that allows gradient flow through BOTH query_latents AND encoder.

    Unlike train_composer.SkillComposer, the encoder is trainable here. Used in
    joint SFT where both query_latents (7K) and model (7B) update together.
    """

    def __init__(self, model: nn.Module, latents_per_skill: int = 2):
        super().__init__()
        self.encoder = model  # Same model used as encoder AND main forward
        self.latents_per_skill = latents_per_skill
        D = model.config.hidden_size
        self.query_latents = nn.Parameter(torch.randn(latents_per_skill, D) * 0.02)
        # NOTE: model is NOT frozen here (caller controls requires_grad)

    def encode_skill(self, skill_token_ids: torch.Tensor) -> torch.Tensor:
        """Encode a single skill with grad on query_latents AND encoder."""
        # Embed lookup (gradient flows back to encoder.embed_tokens)
        text_embeds = self.encoder.get_input_embeddings()(skill_token_ids)
        query = self.query_latents.unsqueeze(0).to(text_embeds.dtype)
        combined = torch.cat([text_embeds, query], dim=1)
        attn_mask = torch.ones(combined.shape[:2], dtype=torch.long, device=combined.device)

        # Use inner Qwen2Model (skip lm_head — we only need hidden states)
        inner = self.encoder.model if hasattr(self.encoder, 'model') else self.encoder
        outputs = inner(
            inputs_embeds=combined,
            attention_mask=attn_mask,
            return_dict=True,
            use_cache=False,
        )
        hidden = outputs.last_hidden_state
        return hidden[:, -self.latents_per_skill:, :]  # (1, k, D)


def train_joint_sft(
    model_path: str, sft_data_path: str, skills_json_path: str,
    output_dir: str, latents_per_skill: int = 2,
    epochs: int = 1, model_lr: float = 1e-5, latent_lr: float = 5e-3,
    max_length: int = 4096, max_samples: int = None,
    device: str = "cuda", seed: int = 42,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    logger.info(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    composer = JointComposer(model, latents_per_skill=latents_per_skill)
    composer.query_latents = nn.Parameter(composer.query_latents.to(device).float())
    logger.info(f"query_latents shape={composer.query_latents.shape}, "
                f"trainable model params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    dataset = ComposerDistillationDataset(sft_data_path, tokenizer, max_length)
    if max_samples:
        dataset.samples = dataset.samples[:max_samples]
    logger.info(f"Using {len(dataset)} samples")

    optimizer = torch.optim.AdamW([
        {"params": [composer.query_latents], "lr": latent_lr, "weight_decay": 0.0},
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": model_lr, "weight_decay": 0.01},
    ])

    logger.info(f"Optimizer: model_lr={model_lr}, latent_lr={latent_lr}, weight_decay model=0.01 latent=0.0")
    logger.info(f"Training {epochs} epochs on {len(dataset)} samples")

    for epoch in range(epochs):
        ep_loss, n_ok = 0.0, 0

        for i, sample in enumerate(dataset):
            instruction, output = sample["instruction"], sample["output"]
            before, skill_texts, after = parse_skills_from_content(instruction)
            if not skill_texts:
                continue

            # Encode each skill (gradient flows through query_latents AND encoder)
            all_latents = []
            for skill_text in skill_texts:
                tokens = tokenizer(
                    skill_text, return_tensors="pt", truncation=True,
                    max_length=128, add_special_tokens=False,
                ).to(device)
                latent = composer.encode_skill(tokens["input_ids"])
                all_latents.append(latent[0])
            latent_tokens = torch.cat(all_latents, dim=0)
            n_lat = latent_tokens.shape[0]

            placeholder = "[LATENT_SKILLS]"
            modified_instruction = before + "\n" + placeholder + "\n" + after
            messages = [
                {"role": "user", "content": modified_instruction},
                {"role": "assistant", "content": output},
            ]
            full_text = tokenizer.apply_chat_template(messages, tokenize=False)
            pos = full_text.find(placeholder)
            if pos < 0:
                continue

            before_text = full_text[:pos]
            after_text = full_text[pos + len(placeholder):]
            before_ids = tokenizer(before_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            after_ids = tokenizer(after_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

            embed = model.get_input_embeddings()
            before_embeds = embed(before_ids)
            after_embeds = embed(after_ids)
            latent_embeds = latent_tokens.unsqueeze(0).to(before_embeds.dtype)
            inputs_embeds = torch.cat([before_embeds, latent_embeds, after_embeds], dim=1)

            if inputs_embeds.shape[1] > max_length:
                inputs_embeds = inputs_embeds[:, :max_length, :]
            total_len = inputs_embeds.shape[1]
            attention_mask = torch.ones(1, total_len, dtype=torch.long, device=device)

            assistant_marker = "<|im_start|>assistant\n"
            n_after_prompt = 0
            if assistant_marker in after_text:
                pre_resp = after_text[:after_text.find(assistant_marker) + len(assistant_marker)]
                n_after_prompt = len(tokenizer(pre_resp, add_special_tokens=False)["input_ids"])
            n_prompt = before_ids.shape[1] + n_lat + n_after_prompt

            full_ids = torch.cat([before_ids, torch.zeros(1, n_lat, dtype=torch.long, device=device), after_ids], dim=1)
            if full_ids.shape[1] > max_length:
                full_ids = full_ids[:, :max_length]
            labels = full_ids.clone()
            labels[:, :n_prompt] = -100
            labels[:, before_ids.shape[1]:before_ids.shape[1] + n_lat] = -100

            outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False)
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            optimizer.zero_grad()
            loss.backward()

            # Sanity: both should be non-zero
            ql_grad = composer.query_latents.grad.norm().item() if composer.query_latents.grad is not None else 0.0
            model_grad = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss += loss.item()
            n_ok += 1

            if (i + 1) % 5 == 0 or i == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} step {i+1}/{len(dataset)} "
                    f"loss={loss.item():.4f} ql_grad={ql_grad:.4e} model_grad={model_grad:.2f} "
                    f"ql_norm={composer.query_latents.norm().item():.3f}"
                )

        avg = ep_loss / max(n_ok, 1)
        logger.info(f"Epoch {epoch+1} avg_loss={avg:.4f} ({n_ok} samples)")

        torch.save({
            "query_latents": composer.query_latents.detach().cpu(),
            "epoch": epoch + 1, "avg_loss": avg, "latents_per_skill": latents_per_skill,
        }, output_dir / f"query_latents_epoch{epoch+1}.pt")
        logger.info(f"Saved query_latents to {output_dir / f'query_latents_epoch{epoch+1}.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--sft-data", required=True)
    parser.add_argument("--skills-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=50, help="POC: cap samples for fast test")
    parser.add_argument("--model-lr", type=float, default=1e-5)
    parser.add_argument("--latent-lr", type=float, default=5e-3)
    parser.add_argument("--latents-per-skill", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    train_joint_sft(
        model_path=args.model_path,
        sft_data_path=args.sft_data,
        skills_json_path=args.skills_json,
        output_dir=args.output_dir,
        latents_per_skill=args.latents_per_skill,
        epochs=args.epochs,
        model_lr=args.model_lr,
        latent_lr=args.latent_lr,
        max_length=args.max_length,
        max_samples=args.max_samples,
        seed=args.seed,
    )
