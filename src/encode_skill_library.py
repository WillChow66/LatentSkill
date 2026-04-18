"""
Offline Latent Skill Library Encoding

Pre-computes latent tokens for all skills in the skill library
using the pretrained model + fixed query_latents. The results are
cached and reused during SFT training (no re-encoding needed).

Output:
  - latent_skill_library.pt: {skill_text_hash: latent_tensor}
  - query_latents.pt: the fixed query_latents used for encoding

Usage:
  python -m src.encode_skill_library \
      --model-name Qwen/Qwen2.5-7B-Instruct \
      --output-dir data/latent_library \
      --latents-per-skill 2
"""

import json
import hashlib
import logging
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

SKILLRL_ROOT = Path(__file__).parent.parent / "SkillRL"


def hash_skill(text: str) -> str:
    """Deterministic hash for a skill text string."""
    return hashlib.md5(text.encode()).hexdigest()


def extract_individual_skills(skills_json_path: str) -> list[dict]:
    """Load and extract all individual skills from SkillRL's skill library."""
    with open(skills_json_path) as f:
        data = json.load(f)

    skills = []
    # General skills
    for s in data.get("general_skills", []):
        text = f"**{s['title']}**: {s['principle']}"
        if "when_to_apply" in s:
            text += f" Apply when: {s['when_to_apply']}"
        skills.append({"id": s["skill_id"], "text": text, "type": "general"})

    # Task-specific skills
    for task_type, task_skills in data.get("task_specific_skills", {}).items():
        for s in task_skills:
            text = f"**{s['title']}**: {s['principle']}"
            if "when_to_apply" in s:
                text += f" Apply when: {s['when_to_apply']}"
            skills.append({"id": s["skill_id"], "text": text, "type": task_type})

    # Common mistakes
    for i, m in enumerate(data.get("common_mistakes", [])):
        if "title" in m and "principle" in m:
            text = f"**{m['title']}**: {m['principle']}"
            skills.append({"id": f"mistake_{i}", "text": text, "type": "mistakes"})

    return skills


@torch.no_grad()
def encode_skill_library(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    skills_json_path: str = None,
    output_dir: str = "data/latent_library",
    latents_per_skill: int = 2,
    device: str = "cuda",
    seed: int = 42,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if skills_json_path is None:
        skills_json_path = str(SKILLRL_ROOT / "memory_data" / "alfworld" / "claude_style_skills.json")

    # Load pretrained model (NOT fine-tuned — use the same model as SFT starting point)
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.eval()

    hidden_size = model.config.hidden_size

    # Create fixed query_latents (random, deterministic with seed)
    torch.manual_seed(seed)
    query_latents = torch.randn(latents_per_skill, hidden_size, dtype=torch.bfloat16) * 0.02
    query_latents = query_latents.to(device)
    logger.info(f"query_latents shape: {query_latents.shape}, seed: {seed}")

    # Load and extract all individual skills
    all_skills = extract_individual_skills(skills_json_path)
    logger.info(f"Total skills to encode: {len(all_skills)}")

    # Encode each skill
    latent_library = {}  # skill_text_hash -> (latents_per_skill, hidden_size)
    skill_index = {}     # skill_text_hash -> {id, text, type}

    for i, skill in enumerate(all_skills):
        skill_text = skill["text"]
        skill_hash = hash_skill(skill_text)

        # Tokenize
        tokens = tokenizer(skill_text, return_tensors="pt", truncation=True,
                           max_length=128, add_special_tokens=False).to(device)
        text_emb = model.get_input_embeddings()(tokens["input_ids"])  # (1, seq_len, D)

        # Append query_latents
        query = query_latents.unsqueeze(0)  # (1, k, D)
        combined = torch.cat([text_emb, query], dim=1)  # (1, seq_len+k, D)

        attn_mask = torch.ones(combined.shape[:2], dtype=torch.long, device=device)

        # Forward pass through pretrained model
        outputs = model(
            inputs_embeds=combined,
            attention_mask=attn_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        # Extract last k hidden states from last layer
        hidden = outputs.hidden_states[-1]
        latent = hidden[0, -latents_per_skill:, :].cpu()  # (k, D)

        latent_library[skill_hash] = latent
        skill_index[skill_hash] = {
            "id": skill["id"],
            "text": skill_text[:100],  # truncated for readability
            "type": skill["type"],
        }

        if (i + 1) % 10 == 0 or i == len(all_skills) - 1:
            logger.info(f"  Encoded {i+1}/{len(all_skills)}: {skill['id']} ({skill['type']})")

    # Save
    torch.save({
        "latent_library": latent_library,       # {hash: (k, D) tensor}
        "skill_index": skill_index,             # {hash: {id, text, type}}
        "query_latents": query_latents.cpu(),   # (k, D)
        "model_name": model_name,
        "latents_per_skill": latents_per_skill,
        "hidden_size": hidden_size,
        "seed": seed,
    }, output_dir / "latent_skill_library.pt")

    logger.info(f"\nSaved to {output_dir / 'latent_skill_library.pt'}")
    logger.info(f"  {len(latent_library)} skills encoded")
    logger.info(f"  Each skill: ({latents_per_skill}, {hidden_size})")
    logger.info(f"  query_latents seed: {seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--skills-json", default=None)
    parser.add_argument("--output-dir", default="data/latent_library")
    parser.add_argument("--latents-per-skill", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    encode_skill_library(
        model_name=args.model_name,
        skills_json_path=args.skills_json,
        output_dir=args.output_dir,
        latents_per_skill=args.latents_per_skill,
        device=args.device,
        seed=args.seed,
    )
