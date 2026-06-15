"""
Expand SFT checkpoint vocabulary with latent skill tokens.

Takes:
  1. SFT checkpoint (full-param trained model)
  2. Pre-computed latent skill library (latent_skill_library.pt)

Does:
  1. Adds special tokens (SKILL_xxx_a, SKILL_xxx_b) to tokenizer
  2. Expands model's embedding matrix
  3. Initializes new token embeddings from pre-computed latent vectors
  4. Saves expanded checkpoint

Also includes a verification mode that checks inputs_embeds ≈ token ID equivalence.

Usage:
  # Expand vocab
  python -m src.expand_vocab_with_skills expand \
      --sft-checkpoint /path/to/sft_checkpoint \
      --latent-library data/latent_library/latent_skill_library.pt \
      --output-dir /path/to/expanded_checkpoint

  # Verify equivalence
  python -m src.expand_vocab_with_skills verify \
      --expanded-checkpoint /path/to/expanded_checkpoint \
      --latent-library data/latent_library/latent_skill_library.pt
"""

import json
import logging
import argparse
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def _suffix_for(j: int) -> str:
    """Map latent index j → token suffix. Uses a, b, c, ... for j<26 then a0,a1,...
    Stable & readable for any k up to 36, and unique for any k via numeric fallback."""
    if j < 26:
        return chr(ord("a") + j)
    return f"x{j}"  # numeric fallback (unlikely to hit; we use k≤8)


def build_skill_token_names(latent_library_data: dict) -> tuple:
    """Build ordered list of special token names from latent library.

    Each skill gets ``latents_per_skill`` tokens: SKILL_{id}_a, SKILL_{id}_b, ...
    Returns (token_names, sorted_hashes).
    """
    skill_index = latent_library_data["skill_index"]
    k = latent_library_data["latents_per_skill"]
    token_names = []

    # Sort by skill_id for deterministic ordering
    sorted_hashes = sorted(skill_index.keys(), key=lambda h: skill_index[h]["id"])

    for h in sorted_hashes:
        skill_id = skill_index[h]["id"]
        for j in range(k):
            token_names.append(f"SKILL_{skill_id}_{_suffix_for(j)}")

    return token_names, sorted_hashes


def expand_vocab(
    sft_checkpoint: str,
    latent_library_path: str,
    output_dir: str,
    device: str = "cpu",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load latent library
    logger.info(f"Loading latent library: {latent_library_path}")
    lib_data = torch.load(latent_library_path, map_location="cpu")
    latent_library = lib_data["latent_library"]
    latents_per_skill = lib_data["latents_per_skill"]
    logger.info(f"  {len(latent_library)} skills, k={latents_per_skill}")

    # Build token names
    token_names, sorted_hashes = build_skill_token_names(lib_data)
    logger.info(f"  {len(token_names)} new tokens to add")

    # Load model and tokenizer
    logger.info(f"Loading SFT checkpoint: {sft_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        sft_checkpoint, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)

    original_vocab_size = len(tokenizer)
    logger.info(f"  Original vocab size: {original_vocab_size}")

    # Add special tokens
    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": token_names,
    })
    logger.info(f"  Added {num_added} special tokens")

    # Expand embedding matrix
    model.resize_token_embeddings(len(tokenizer))
    new_vocab_size = len(tokenizer)
    logger.info(f"  New vocab size: {new_vocab_size}")

    # Untie LM head from input embeddings
    # This allows SKILL tokens to have different input vs output representations:
    #   Input embedding = latent vector (model can "read" skills)
    #   LM head weight  = zero (model won't "generate" SKILL tokens)
    if model.config.tie_word_embeddings:
        logger.info("  Untying word embeddings from LM head")
        model.config.tie_word_embeddings = False
        # After untying, lm_head.weight is a separate copy
        if hasattr(model, 'lm_head'):
            model.lm_head.weight = nn.Parameter(model.lm_head.weight.data.clone())

    # Initialize new token embeddings from latent library
    embed_weight = model.get_input_embeddings().weight.data
    lm_head_weight = model.lm_head.weight.data if hasattr(model, 'lm_head') else None

    for i, h in enumerate(sorted_hashes):
        latent_vecs = latent_library[h]  # (k, D)
        skill_id = lib_data["skill_index"][h]["id"]

        for j in range(latents_per_skill):
            token_name = f"SKILL_{skill_id}_{_suffix_for(j)}"
            token_id = tokenizer.convert_tokens_to_ids(token_name)

            if token_id == tokenizer.unk_token_id:
                logger.warning(f"  Token {token_name} not found in tokenizer!")
                continue

            # Input embedding = latent vector (model reads SKILL tokens)
            embed_weight[token_id] = latent_vecs[j].to(embed_weight.dtype)

            # LM head weight = ZERO (model never generates SKILL tokens)
            if lm_head_weight is not None:
                lm_head_weight[token_id] = 0.0

    logger.info(f"  Initialized {len(sorted_hashes) * latents_per_skill} token embeddings")
    logger.info(f"  LM head weights for SKILL tokens set to ZERO (untied)")

    # Save expanded checkpoint
    logger.info(f"Saving to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save skill token mapping for later use
    skill_token_map = {}
    for h in sorted_hashes:
        skill_id = lib_data["skill_index"][h]["id"]
        skill_type = lib_data["skill_index"][h]["type"]
        tokens = [f"SKILL_{skill_id}_{_suffix_for(j)}" for j in range(latents_per_skill)]
        token_ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
        skill_token_map[skill_id] = {
            "hash": h,
            "type": skill_type,
            "tokens": tokens,
            "token_ids": token_ids,
        }

    with open(output_dir / "skill_token_map.json", "w") as f:
        json.dump(skill_token_map, f, indent=2)
    logger.info(f"  Saved skill_token_map.json ({len(skill_token_map)} skills)")

    logger.info("Done!")
    return skill_token_map


@torch.no_grad()
def verify_equivalence(
    expanded_checkpoint: str,
    latent_library_path: str,
    device: str = "cuda",
):
    """Verify that token ID lookup ≈ inputs_embeds injection."""
    logger.info("=== Verifying embedding equivalence ===")

    # Load expanded model
    tokenizer = AutoTokenizer.from_pretrained(expanded_checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        expanded_checkpoint, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.eval()

    # Load latent library
    lib_data = torch.load(latent_library_path, map_location="cpu")
    latent_library = lib_data["latent_library"]
    _, sorted_hashes = build_skill_token_names(lib_data)

    # Load skill token map
    with open(Path(expanded_checkpoint) / "skill_token_map.json") as f:
        skill_token_map = json.load(f)

    # Test prompt
    test_prompt = "Your task is to: clean a mug."
    test_ids = tokenizer.encode(test_prompt, return_tensors="pt").to(device)

    # Pick first skill for testing
    first_skill_id = list(skill_token_map.keys())[0]
    first_skill = skill_token_map[first_skill_id]
    first_hash = first_skill["hash"]
    latent_vecs = latent_library[first_hash].to(device).to(torch.bfloat16)

    # Method 1: inputs_embeds
    text_emb = model.get_input_embeddings()(test_ids)
    latent_emb = latent_vecs.unsqueeze(0)  # (1, 2, D)
    full_emb = torch.cat([text_emb, latent_emb], dim=1)
    out1 = model(inputs_embeds=full_emb)
    logits1 = out1.logits

    # Method 2: token IDs
    skill_token_ids = torch.tensor([first_skill["token_ids"]], device=device)
    full_ids = torch.cat([test_ids, skill_token_ids], dim=1)
    out2 = model(input_ids=full_ids)
    logits2 = out2.logits

    # Compare
    diff = (logits1 - logits2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    logger.info(f"  Test prompt: '{test_prompt}'")
    logger.info(f"  Skill: {first_skill_id} ({first_skill['tokens']})")
    logger.info(f"  Max logit difference: {max_diff:.8f}")
    logger.info(f"  Mean logit difference: {mean_diff:.8f}")

    if max_diff < 1e-3:
        logger.info("  ✅ PASSED: inputs_embeds ≈ token ID lookup")
    else:
        logger.warning(f"  ⚠️ DIFFERENCE DETECTED: max_diff={max_diff}")

    return max_diff


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # expand command
    expand_parser = subparsers.add_parser("expand")
    expand_parser.add_argument("--sft-checkpoint", required=True)
    expand_parser.add_argument("--latent-library", required=True)
    expand_parser.add_argument("--output-dir", required=True)

    # verify command
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--expanded-checkpoint", required=True)
    verify_parser.add_argument("--latent-library", required=True)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.command == "expand":
        expand_vocab(args.sft_checkpoint, args.latent_library, args.output_dir)
    elif args.command == "verify":
        verify_equivalence(args.expanded_checkpoint, args.latent_library)
    else:
        parser.print_help()
