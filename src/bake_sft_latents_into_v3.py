"""Bake Stage-1 SFT-distilled latents into v2 checkpoint → v3.

Stage 1 trained query_latents so that latent SKILL tokens produce the same agent
behavior as text skills. Stage 2 (full-param RL on the actor) needs those
latents baked into the model's embed_tokens at the SKILL token positions, so
the Composer is no longer in the RL loop.

Flow:
  1. Load v2 ckpt (has 88 SKILL tokens in vocab, lm_head[SKILL]=0)
  2. Load Stage 1 trained query_latents
  3. Run Composer (v2 as encoder, frozen) on 44 skill texts -> 88 latent vecs
  4. Overwrite v2.embed_tokens.weight[SKILL_token_ids] with these vecs
  5. Save as v3 ckpt

Usage:
  python -m src.bake_sft_latents_into_v3 \
      --v2-ckpt /work/.../latent_skills_token_v2 \
      --query-latents data/debug_composer/trained_query_latents.pt \
      --output-dir /work/.../latent_skills_token_v3
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.skill_composer import SkillComposer

logger = logging.getLogger(__name__)

SKILLRL_ROOT = Path(__file__).parent.parent / "SkillRL"


@torch.no_grad()
def bake(v2_ckpt: str, query_latents_path: str, output_dir: str,
         skills_json_path: str = None, skill_token_map_path: str = None,
         latents_per_skill: int = 2, device: str = "cuda"):
    if skills_json_path is None:
        skills_json_path = str(SKILLRL_ROOT / "memory_data" / "alfworld" / "claude_style_skills.json")
    if skill_token_map_path is None:
        skill_token_map_path = str(Path(v2_ckpt) / "skill_token_map.json")

    logger.info(f"Loading v2 model: {v2_ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(v2_ckpt, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        v2_ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()

    composer = SkillComposer(encoder_model=model, latents_per_skill=latents_per_skill)
    composer = composer.to(device)
    composer.load_pretrained(query_latents_path)
    logger.info(f"query_latents shape: {composer.query_latents.shape}, "
                f"norm: {composer.query_latents.norm().item():.4f}")

    with open(skill_token_map_path) as f:
        skill_token_map = json.load(f)
    composer.setup_skills(skills_json_path, tokenizer, skill_token_map)

    logger.info(f"Encoding {len(composer._skill_texts)} skills...")
    latent_map = composer.encode_all_skills()
    logger.info(f"Got {len(latent_map)} latent vectors (token_id -> vec)")

    embed = model.get_input_embeddings()
    orig_dtype = embed.weight.dtype
    n_written = 0
    for token_id, latent_vec in latent_map.items():
        embed.weight.data[token_id] = latent_vec.to(orig_dtype)
        n_written += 1
    logger.info(f"Overwrote {n_written} SKILL token embeddings in v2.embed_tokens")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving v3 model to {out}")
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    src_map = Path(skill_token_map_path)
    dst_map = out / "skill_token_map.json"
    if src_map.exists() and not dst_map.exists():
        dst_map.write_text(src_map.read_text())
    logger.info(f"Done. v3 at {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-ckpt", required=True)
    parser.add_argument("--query-latents", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skills-json", default=None)
    parser.add_argument("--skill-token-map", default=None)
    parser.add_argument("--latents-per-skill", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    bake(
        v2_ckpt=args.v2_ckpt,
        query_latents_path=args.query_latents,
        output_dir=args.output_dir,
        skills_json_path=args.skills_json,
        skill_token_map_path=args.skill_token_map,
        latents_per_skill=args.latents_per_skill,
    )
