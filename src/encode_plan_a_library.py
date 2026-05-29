"""Encode skill library for a Plan A SFT ckpt (no SKILL tokens in vocab).

Used by eval to convert 44 text skills into latent vectors using
the trained Composer (model + query_latents). Output is a hash-keyed
dict compatible with eval_latent_alfworld.py.

Usage:
  python -m src.encode_plan_a_library \
      --model /work/.../joint_sft_A/model_epoch4 \
      --query-latents /work/.../joint_sft_A/query_latents_step7484.pt \
      --output data/latent_library/plan_a_A_library.pt
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.skill_composer import SkillComposer, hash_skill, extract_individual_skills

logger = logging.getLogger(__name__)
SKILLRL_ROOT = Path(__file__).parent.parent / "SkillRL"


@torch.no_grad()
def encode(model_path: str, query_latents_path: str, output_path: str,
           skills_json_path: str = None, latents_per_skill: int = 2,
           device: str = "cuda"):
    if skills_json_path is None:
        skills_json_path = str(SKILLRL_ROOT / "memory_data" / "alfworld" / "claude_style_skills.json")

    logger.info(f"Loading encoder model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoder = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    composer = SkillComposer(encoder_model=encoder, latents_per_skill=latents_per_skill).to(device)
    composer.load_pretrained(query_latents_path)
    composer._tokenizer = tokenizer
    logger.info(f"query_latents shape: {composer.query_latents.shape}, "
                f"norm: {composer.query_latents.norm().item():.4f}")

    all_skills = extract_individual_skills(skills_json_path)
    skill_texts = [s["text"] for s in all_skills]
    logger.info(f"Encoding {len(skill_texts)} skills...")

    all_latents = composer.encode_skills(skill_texts)  # (n_skills, k, D)

    latent_library = {}
    skill_index = {}
    for i, s in enumerate(all_skills):
        h = hash_skill(s["text"])
        latent_library[h] = all_latents[i].cpu()
        skill_index[h] = {"id": s["id"], "type": s["type"], "text": s["text"][:100]}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "latent_library": latent_library,
        "skill_index": skill_index,
        "query_latents": composer.query_latents.detach().cpu(),
        "model_path": model_path,
        "query_latents_source": query_latents_path,
        "latents_per_skill": latents_per_skill,
        "hidden_size": composer.hidden_size,
    }, output_path)

    logger.info(f"Saved {len(latent_library)} skills to {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--query-latents", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--skills-json", default=None)
    p.add_argument("--latents-per-skill", type=int, default=2)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    encode(args.model, args.query_latents, args.output, args.skills_json,
           args.latents_per_skill)
