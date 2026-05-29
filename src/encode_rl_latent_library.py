"""Re-encode skill library using RL-trained query_latents.

Uses Composer with v2 base + RL's best query_latents to encode all 44 skills.
Output is compatible with eval_latent_alfworld.py (same hash-keyed dict format).

Usage:
  python -m src.encode_rl_latent_library \
      --model /work/nvme/bdns/xzhou10/checkpoints/latent_skills_token_v2 \
      --query-latents /work/nvme/bfdz/xzhou10/checkpoints/rl_e2e_full/best/query_latents_step40.pt \
      --output data/latent_library/rl_best_latent_library.pt
"""

import argparse
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
           skills_json_path: str = None, skill_token_map_path: str = None,
           latents_per_skill: int = 2, device: str = "cuda"):
    if skills_json_path is None:
        skills_json_path = str(SKILLRL_ROOT / "memory_data" / "alfworld" / "claude_style_skills.json")
    if skill_token_map_path is None:
        skill_token_map_path = str(Path(model_path) / "skill_token_map.json")

    logger.info(f"Loading encoder model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    encoder = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    composer = SkillComposer(encoder_model=encoder, latents_per_skill=latents_per_skill)
    composer = composer.to(device)
    composer.load_pretrained(query_latents_path)
    logger.info(f"query_latents shape: {composer.query_latents.shape}, "
                f"norm: {composer.query_latents.norm().item():.4f}")

    with open(skill_token_map_path) as f:
        skill_token_map = json.load(f)
    composer.setup_skills(skills_json_path, tokenizer, skill_token_map)

    all_skills = extract_individual_skills(skills_json_path)
    logger.info(f"Encoding {len(all_skills)} skills...")

    latent_library = {}  # hash -> (k, D) tensor
    skill_index = {}

    skill_texts = composer._skill_texts
    all_latents = composer.encode_skills(skill_texts)  # (n, k, D)

    skill_id_to_meta = {s["id"]: s for s in all_skills}
    for i, skill_text in enumerate(skill_texts):
        h = hash_skill(skill_text)
        latent_library[h] = all_latents[i].cpu()  # (k, D)
        meta_id = None
        for s in all_skills:
            if s["text"] == skill_text:
                meta_id = s["id"]
                skill_index[h] = {"id": s["id"], "text": skill_text[:100], "type": s["type"]}
                break

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
    logger.info(f"  Each skill: ({latents_per_skill}, {composer.hidden_size})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Encoder model path (v2 checkpoint)")
    parser.add_argument("--query-latents", required=True, help="Trained query_latents .pt")
    parser.add_argument("--output", required=True, help="Output latent library .pt")
    parser.add_argument("--skills-json", default=None)
    parser.add_argument("--skill-token-map", default=None)
    parser.add_argument("--latents-per-skill", type=int, default=2)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    encode(
        model_path=args.model,
        query_latents_path=args.query_latents,
        output_path=args.output,
        skills_json_path=args.skills_json,
        skill_token_map_path=args.skill_token_map,
        latents_per_skill=args.latents_per_skill,
    )
