"""Build v4 ckpt = v3 + 200 spare SKILL tokens for dynamic skill bank.

v3 has 88 static SKILL tokens (cle/pic/... × 44 skills × 2 latents each) baked
with Stage-1 SFT latents.

v4 extends v3's vocab with 200 additional spare SKILL tokens reserved for
`dyn_001_a ... dyn_100_b` — i.e. 100 dynamically-added skill slots. When
SkillUpdater generates new skills during RL training, the trainer assigns each
new skill to the next free dyn slot, encodes its text via the frozen Composer,
and writes the latent into v4.embed_tokens at that slot's token ids.

Spare slots:
  - embed_tokens.weight[spare_ids] = 0 (filled at runtime)
  - lm_head.weight[spare_ids] = 0    (model shouldn't generate them)
  - tokenizer has "SKILL_dyn_NNN_a" / "SKILL_dyn_NNN_b" strings
  - skill_token_map.json gains 100 "dyn_NNN" placeholders with hash=None

Usage:
  python -m src.bake_sft_latents_into_v4 \
      --v3-ckpt /work/.../latent_skills_token_v3 \
      --output-dir /work/.../latent_skills_token_v4 \
      --num-dyn-slots 100
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def build_v4(v3_ckpt: str, output_dir: str, num_dyn_slots: int = 100,
             latents_per_skill: int = 2, device: str = "cpu"):
    v3_path = Path(v3_ckpt)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading v3 tokenizer + model: {v3_path}")
    tokenizer = AutoTokenizer.from_pretrained(v3_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        v3_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()

    old_vocab_size = model.config.vocab_size
    logger.info(f"v3 vocab size: {old_vocab_size}")

    # Build spare dyn SKILL token strings
    num_new_tokens = num_dyn_slots * latents_per_skill
    new_skill_tokens = []
    for n in range(1, num_dyn_slots + 1):
        for suffix in ["a", "b"] if latents_per_skill == 2 else [chr(ord('a') + i) for i in range(latents_per_skill)]:
            new_skill_tokens.append(f"SKILL_dyn_{n:03d}_{suffix}")

    assert len(new_skill_tokens) == num_new_tokens

    num_added = tokenizer.add_tokens(new_skill_tokens, special_tokens=False)
    logger.info(f"Added {num_added} new dyn SKILL tokens to tokenizer")
    assert num_added == num_new_tokens, f"Expected {num_new_tokens} new, got {num_added}"

    new_vocab_size = len(tokenizer)
    logger.info(f"v4 vocab size: {new_vocab_size}")

    # Resize model embeddings + lm_head
    model.resize_token_embeddings(new_vocab_size)

    embed = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()

    # Zero-init new rows (spare slots, filled by Composer at runtime)
    with torch.no_grad():
        embed.weight.data[old_vocab_size:new_vocab_size].zero_()
        lm_head.weight.data[old_vocab_size:new_vocab_size].zero_()

    logger.info(f"Zeroed {num_new_tokens} new rows in embed_tokens and lm_head")

    # Verify: record token_ids for each dyn slot (k-generic: ALL latents_per_skill
    # suffixes, not just a/b — k=8 was silently dropping 6 of 8 token rows per slot).
    suffixes = ["a", "b"] if latents_per_skill == 2 else [chr(ord('a') + i) for i in range(latents_per_skill)]
    dyn_slot_map = {}
    for n in range(1, num_dyn_slots + 1):
        slot_id = f"dyn_{n:03d}"
        toks = [f"SKILL_dyn_{n:03d}_{s}" for s in suffixes]
        ids = [tokenizer.convert_tokens_to_ids(t) for t in toks]
        dyn_slot_map[slot_id] = {
            "tokens": toks,
            "token_ids": ids,
            "hash": None,  # filled when a dyn skill is assigned
            "type": "general",
            "text": None,  # filled when a dyn skill is assigned
        }

    # Save model + tokenizer
    logger.info(f"Saving v4 to {out_path}")
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)

    # Extend skill_token_map.json: keep static entries, append dyn slots
    old_map_path = v3_path / "skill_token_map.json"
    with open(old_map_path) as f:
        skill_token_map = json.load(f)
    n_static = len(skill_token_map)
    skill_token_map.update(dyn_slot_map)
    new_map_path = out_path / "skill_token_map.json"
    with open(new_map_path, "w") as f:
        json.dump(skill_token_map, f, indent=2)

    logger.info(f"Wrote extended skill_token_map.json: {n_static} static + "
                f"{num_dyn_slots} dyn slots = {len(skill_token_map)} entries")
    logger.info(f"Done. v4 at {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-ckpt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-dyn-slots", type=int, default=100)
    parser.add_argument("--latents-per-skill", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    build_v4(
        v3_ckpt=args.v3_ckpt,
        output_dir=args.output_dir,
        num_dyn_slots=args.num_dyn_slots,
        latents_per_skill=args.latents_per_skill,
    )
