"""
Stage 1: Train Skill Composer

Trains learnable query_latents to compress text skills into latent tokens
via distillation. The agent model is frozen; only query_latents are trained.

Architecture (following LatentMem):
  - Encoder = agent model itself (frozen)
  - Learnable query_latents appended to skill text embeddings
  - Forward through encoder → extract last hidden states at query positions
  - These become the latent skill tokens

Training objective:
  - Replace text skills in SFT prompts with encoded latent tokens
  - Compute cross-entropy loss on agent response (distillation)
  - Backprop updates only query_latents

Usage:
  python -m src.train_composer \
      --model-path /path/to/sft_checkpoint \
      --sft-data /path/to/sft_data.parquet \
      --skills-json SkillRL/memory_data/alfworld/claude_style_skills.json \
      --output-dir data/trained_composer \
      --latents-per-skill 2 \
      --epochs 10 \
      --lr 5e-3
"""

import json
import hashlib
import logging
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

SKILLS_START_MARKER = "## Retrieved Relevant Experience"
SKILLS_END_MARKER = "## Current Progress"


def hash_skill(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def extract_individual_skills(skills_json_path: str) -> list[dict]:
    """Load all skills from SkillRL's skill library."""
    with open(skills_json_path) as f:
        data = json.load(f)

    skills = []
    for s in data.get("general_skills", []):
        text = f"**{s['title']}**: {s['principle']}"
        if "when_to_apply" in s:
            text += f" Apply when: {s['when_to_apply']}"
        skills.append({"id": s["skill_id"], "text": text, "type": "general"})

    for task_type, task_skills in data.get("task_specific_skills", {}).items():
        for s in task_skills:
            text = f"**{s['title']}**: {s['principle']}"
            if "when_to_apply" in s:
                text += f" Apply when: {s['when_to_apply']}"
            skills.append({"id": s["skill_id"], "text": text, "type": task_type})

    for i, m in enumerate(data.get("common_mistakes", [])):
        if "title" in m and "principle" in m:
            text = f"**{m['title']}**: {m['principle']}"
            skills.append({"id": f"mistake_{i}", "text": text, "type": "mistakes"})

    return skills


class SkillComposer(nn.Module):
    """Compresses text skills into latent tokens using the agent model as encoder.

    Following LatentMem's architecture:
    - Appends learnable query_latents to skill text embeddings
    - Forward through frozen encoder → extract last hidden states
    """

    def __init__(self, encoder_model: nn.Module, latents_per_skill: int = 2):
        super().__init__()
        self.encoder = encoder_model
        self.latents_per_skill = latents_per_skill
        hidden_size = encoder_model.config.hidden_size

        # Learnable query latents (shared across all skills)
        self.query_latents = nn.Parameter(
            torch.randn(latents_per_skill, hidden_size) * 0.02
        )

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

    def encode_skill(self, skill_token_ids: torch.Tensor) -> torch.Tensor:
        """Encode a single skill text into latent tokens.

        Args:
            skill_token_ids: (1, seq_len) token IDs for one skill text

        Returns:
            latent_tokens: (1, k, D) latent representations
        """
        # Get text embeddings from frozen encoder
        with torch.no_grad():
            text_embeds = self.encoder.get_input_embeddings()(skill_token_ids)

        # Append learnable query_latents
        query = self.query_latents.unsqueeze(0).to(text_embeds.dtype)  # (1, k, D)
        combined = torch.cat([text_embeds, query], dim=1)  # (1, seq_len+k, D)

        attn_mask = torch.ones(combined.shape[:2], dtype=torch.long, device=combined.device)

        # Forward through frozen encoder (but gradients flow through query_latents)
        outputs = self.encoder(
            inputs_embeds=combined,
            attention_mask=attn_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        # Extract last k hidden states (at query_latent positions)
        hidden = outputs.hidden_states[-1]
        latent_tokens = hidden[:, -self.latents_per_skill:, :]  # (1, k, D)

        return latent_tokens

    def encode_all_skills(self, skill_texts: list[str], tokenizer) -> dict:
        """Encode all skills and return a latent library.

        Returns:
            {skill_hash: (k, D) tensor}
        """
        latent_library = {}
        device = self.query_latents.device

        for skill_text in skill_texts:
            tokens = tokenizer(
                skill_text, return_tensors="pt", truncation=True,
                max_length=128, add_special_tokens=False
            ).to(device)

            latent = self.encode_skill(tokens["input_ids"])  # (1, k, D)
            skill_hash = hash_skill(skill_text)
            latent_library[skill_hash] = latent[0].detach()  # (k, D)

        return latent_library


class ComposerDistillationDataset(Dataset):
    """Dataset for Composer distillation training.

    Supports SkillRL's SFT data format: columns 'instruction' + 'output'.
    Each instruction contains text skills under "## Retrieved Relevant Experience".
    """

    def __init__(self, parquet_path: str, tokenizer, max_length: int = 4096):
        df = pd.read_parquet(parquet_path)
        self.samples = []

        for _, row in df.iterrows():
            # SkillRL SFT data format
            instruction = row.get("instruction", row.get("prompt", None))
            output = row.get("output", row.get("answer", None))

            if instruction is None or output is None:
                continue
            instruction = str(instruction)
            output = str(output)

            # Check if skills are present
            if SKILLS_START_MARKER not in instruction or SKILLS_END_MARKER not in instruction:
                continue

            self.samples.append({
                "instruction": instruction,
                "output": output,
            })

        logger.info(f"Loaded {len(self.samples)} samples from {parquet_path}")

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def parse_skills_from_content(content: str) -> tuple[str, list[str], str]:
    """Parse a message content to extract individual skill texts.

    Returns:
        (before_skills, list_of_skill_texts, after_skills)
    """
    start = content.find(SKILLS_START_MARKER)
    end = content.find(SKILLS_END_MARKER)
    if start == -1 or end == -1:
        return content, [], ""

    before = content[:start + len(SKILLS_START_MARKER)]
    skills_section = content[start + len(SKILLS_START_MARKER):end]
    after = content[end:]

    individual_skills = []
    for line in skills_section.strip().split("\n"):
        line = line.strip()
        if line.startswith("- **"):
            individual_skills.append(line[2:])  # Remove "- " prefix

    return before, individual_skills, after


def train_composer(
    model_path: str,
    sft_data_path: str,
    skills_json_path: str,
    output_dir: str,
    latents_per_skill: int = 2,
    epochs: int = 10,
    lr: float = 5e-3,
    batch_size: int = 1,
    max_length: int = 4096,
    device: str = "cuda",
    seed: int = 42,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)

    # Load agent model (frozen, used as encoder)
    logger.info(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.eval()

    # Create Composer and move query_latents to device
    composer = SkillComposer(model, latents_per_skill=latents_per_skill)
    composer.query_latents = nn.Parameter(composer.query_latents.to(device))
    logger.info(f"Composer created: query_latents shape = {composer.query_latents.shape}, device = {composer.query_latents.device}")
    logger.info(f"Trainable params: {sum(p.numel() for p in composer.parameters() if p.requires_grad):,}")

    # Load all skills
    all_skills = extract_individual_skills(skills_json_path)
    skill_text_to_hash = {s["text"]: hash_skill(s["text"]) for s in all_skills}
    skill_hash_to_text = {v: k for k, v in skill_text_to_hash.items()}
    logger.info(f"Loaded {len(all_skills)} skills")

    # Load SFT data
    dataset = ComposerDistillationDataset(sft_data_path, tokenizer, max_length)

    # Optimizer (only query_latents) — aligned with verl/SkillRL SFT defaults
    optimizer = torch.optim.AdamW(
        [composer.query_latents], lr=lr, weight_decay=0.0,
        betas=(0.9, 0.95),  # verl/SkillRL SFT default
    )

    # Cosine LR schedule with warmup (verl/SkillRL SFT default warmup_ratio=0.1)
    from transformers import get_cosine_schedule_with_warmup
    total_steps = epochs * len(dataset)
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # Training loop
    logger.info(f"Training for {epochs} epochs, lr={lr}, {len(dataset)} samples; "
                f"betas=(0.9, 0.95), cosine schedule, warmup={warmup_steps}/{total_steps} steps")

    for epoch in range(epochs):
        total_loss = 0.0
        n_samples = 0

        for i, sample in enumerate(dataset):
            instruction = sample["instruction"]
            output = sample["output"]

            # 1. Parse skills from instruction
            before, skill_texts, after = parse_skills_from_content(instruction)

            if not skill_texts:
                continue

            # 2. Encode each skill with the Composer
            all_latents = []
            for skill_text in skill_texts:
                tokens = tokenizer(
                    skill_text, return_tensors="pt", truncation=True,
                    max_length=128, add_special_tokens=False
                ).to(device)
                latent = composer.encode_skill(tokens["input_ids"])  # (1, k, D)
                all_latents.append(latent[0])  # (k, D)

            if not all_latents:
                continue

            latent_tokens = torch.cat(all_latents, dim=0)  # (n_skills * k, D)
            n_latent = latent_tokens.shape[0]

            # 3. Build prompt with latent tokens replacing text skills
            placeholder = "[LATENT_SKILLS]"
            modified_instruction = before + "\n" + placeholder + "\n" + after

            # Apply chat template
            messages = [
                {"role": "user", "content": modified_instruction},
                {"role": "assistant", "content": output},
            ]
            full_text = tokenizer.apply_chat_template(messages, tokenize=False)

            # 4. Split at placeholder and tokenize each part
            placeholder_pos = full_text.find(placeholder)
            if placeholder_pos == -1:
                continue

            before_text = full_text[:placeholder_pos]
            after_text = full_text[placeholder_pos + len(placeholder):]

            before_ids = tokenizer(before_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            after_ids = tokenizer(after_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

            # 5. Build inputs_embeds: [before | latent_tokens | after]
            with torch.no_grad():
                embed_layer = model.get_input_embeddings()
                before_embeds = embed_layer(before_ids)
                after_embeds = embed_layer(after_ids)

            latent_embeds = latent_tokens.unsqueeze(0).to(before_embeds.dtype)
            inputs_embeds = torch.cat([before_embeds, latent_embeds, after_embeds], dim=1)

            if inputs_embeds.shape[1] > max_length:
                inputs_embeds = inputs_embeds[:, :max_length, :]

            total_len = inputs_embeds.shape[1]
            attention_mask = torch.ones(1, total_len, dtype=torch.long, device=device)

            # 6. Build labels: -100 for prompt + latent positions, token IDs for response
            # Find where the response starts (after "<|im_start|>assistant\n")
            assistant_marker = "<|im_start|>assistant\n"
            response_pos = after_text.find(assistant_marker)
            if response_pos == -1:
                # response is in after_text, find it
                response_pos = 0

            # Number of after_ids tokens that are prompt (before response)
            text_before_response_in_after = after_text[:after_text.find(assistant_marker) + len(assistant_marker)] if assistant_marker in after_text else ""
            n_after_prompt_tokens = len(tokenizer(text_before_response_in_after, add_special_tokens=False)["input_ids"]) if text_before_response_in_after else 0
            n_prompt_total = before_ids.shape[1] + n_latent + n_after_prompt_tokens

            # Labels = full token sequence
            full_token_ids = torch.cat([
                before_ids,
                torch.zeros(1, n_latent, dtype=torch.long, device=device),
                after_ids
            ], dim=1)
            if full_token_ids.shape[1] > max_length:
                full_token_ids = full_token_ids[:, :max_length]

            labels = full_token_ids.clone()
            labels[:, :n_prompt_total] = -100
            # Latent positions always masked
            before_len = before_ids.shape[1]
            labels[:, before_len:before_len + n_latent] = -100

            # 7. Forward pass through frozen agent model
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )

            # 8. Cross-entropy loss on response tokens
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # 9. Backward + update
            optimizer.zero_grad()
            loss.backward()

            # Log gradient norm for monitoring
            grad_norm = composer.query_latents.grad.norm().item() if composer.query_latents.grad is not None else 0.0

            torch.nn.utils.clip_grad_norm_([composer.query_latents], 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_samples += 1

            if (i + 1) % 50 == 0:
                avg_loss = total_loss / max(n_samples, 1)
                logger.info(f"  Epoch {epoch+1}/{epochs}, step {i+1}/{len(dataset)}, loss={avg_loss:.4f}, grad_norm={grad_norm:.4f}")

        avg_loss = total_loss / max(n_samples, 1)
        logger.info(f"Epoch {epoch+1}/{epochs}: avg_loss={avg_loss:.4f}, samples={n_samples}")

        # Save checkpoint after each epoch
        torch.save({
            "query_latents": composer.query_latents.detach().cpu(),
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "latents_per_skill": latents_per_skill,
            "hidden_size": model.config.hidden_size,
        }, output_dir / f"checkpoint_epoch{epoch+1}.pt")
        logger.info(f"  Saved checkpoint to {output_dir / f'checkpoint_epoch{epoch+1}.pt'}")

    # Save trained query_latents
    torch.save({
        "query_latents": composer.query_latents.detach().cpu(),
        "latents_per_skill": latents_per_skill,
        "hidden_size": model.config.hidden_size,
        "model_path": model_path,
        "epochs": epochs,
        "lr": lr,
    }, output_dir / "trained_query_latents.pt")
    logger.info(f"Saved trained query_latents to {output_dir / 'trained_query_latents.pt'}")

    # Encode all skills with trained Composer and save library
    logger.info("Encoding all skills with trained Composer...")
    with torch.no_grad():
        # Need gradients through query_latents for encoding
        pass

    latent_library = {}
    skill_index = {}

    for skill in all_skills:
        tokens = tokenizer(
            skill["text"], return_tensors="pt", truncation=True,
            max_length=128, add_special_tokens=False
        ).to(device)

        with torch.no_grad():
            text_embeds = model.get_input_embeddings()(tokens["input_ids"])
            query = composer.query_latents.unsqueeze(0).to(text_embeds.dtype)
            combined = torch.cat([text_embeds, query], dim=1)
            attn_mask = torch.ones(combined.shape[:2], dtype=torch.long, device=device)

            outputs = model(
                inputs_embeds=combined,
                attention_mask=attn_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hidden = outputs.hidden_states[-1]
            latent = hidden[0, -latents_per_skill:, :].cpu()

        skill_hash = hash_skill(skill["text"])
        latent_library[skill_hash] = latent
        skill_index[skill_hash] = {
            "id": skill["id"],
            "text": skill["text"][:100],
            "type": skill["type"],
        }

    torch.save({
        "latent_library": latent_library,
        "skill_index": skill_index,
        "query_latents": composer.query_latents.detach().cpu(),
        "model_path": model_path,
        "latents_per_skill": latents_per_skill,
        "hidden_size": model.config.hidden_size,
    }, output_dir / "latent_skill_library.pt")
    logger.info(f"Saved latent_skill_library.pt with {len(latent_library)} skills")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to SFT checkpoint")
    parser.add_argument("--sft-data", required=True, help="Path to SFT parquet data")
    parser.add_argument("--skills-json", default="SkillRL/memory_data/alfworld/claude_style_skills.json")
    parser.add_argument("--output-dir", default="data/trained_composer")
    parser.add_argument("--latents-per-skill", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    train_composer(
        model_path=args.model_path,
        sft_data_path=args.sft_data,
        skills_json_path=args.skills_json,
        output_dir=args.output_dir,
        latents_per_skill=args.latents_per_skill,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        seed=args.seed,
    )
