"""Stage 2 — Encode the 44 ALFWorld base skills into latent vectors.

Two variants:
  TRAINED   — load Stage 1's Composer ckpt + query_latents + latent_proj
  UNTRAINED — load raw Qwen2.5-3B-Instruct + RANDOM query_latents + RANDOM latent_proj
              (same init seed as Stage 1 → fair control)

Each skill → K latent vectors in Actor's hidden_size (post-projection),
saved in the format expected by expand_vocab_with_skills.py:

  {
    "latent_library": {hash: (k, D_actor) tensor},
    "skill_index":    {hash: {id, text, type}},
    "query_latents":  (k, D_composer) tensor,
    "model_name":     str,
    "latents_per_skill": int,
    "hidden_size":    D_actor,
    "seed":           int,
  }

Usage:
  TRAINED:
  python -m src.encode_latent_library \
      --composer-ckpt /work/.../composer_sft/composer_epoch4 \
      --query-latents /work/.../composer_sft/query_latents_step{N}.pt \
      --actor-path Qwen/Qwen2.5-7B-Instruct \
      --output-dir /work/.../latent_lib_trained \
      --variant TRAINED

  UNTRAINED:
  python -m src.encode_latent_library \
      --composer-ckpt Qwen/Qwen2.5-3B-Instruct \
      --actor-path Qwen/Qwen2.5-7B-Instruct \
      --output-dir /work/.../latent_lib_untrained \
      --variant UNTRAINED \
      --seed 42  # for reproducible random ql + proj
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

SKILLRL_ROOT = Path(__file__).parent.parent / "SkillRL"


def hash_skill(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def extract_individual_skills(skills_json_path: str) -> list:
    with open(skills_json_path) as f:
        data = json.load(f)
    out = []
    for s in data.get("general_skills", []):
        text = f"**{s['title']}**: {s['principle']}"
        if "when_to_apply" in s:
            text += f" Apply when: {s['when_to_apply']}"
        out.append({"id": s["skill_id"], "text": text, "type": "general"})
    for task_type, task_skills in data.get("task_specific_skills", {}).items():
        for s in task_skills:
            text = f"**{s['title']}**: {s['principle']}"
            if "when_to_apply" in s:
                text += f" Apply when: {s['when_to_apply']}"
            out.append({"id": s["skill_id"], "text": text, "type": task_type})
    for i, m in enumerate(data.get("common_mistakes", [])):
        if "title" in m and "principle" in m:
            text = f"**{m['title']}**: {m['principle']}"
            out.append({"id": f"mistake_{i}", "text": text, "type": "mistakes"})
    return out


@torch.no_grad()
def encode_one_skill(composer_inner, query_latents, k, skill_text, tokenizer,
                     pad_token_id, max_length=128):
    """Encode a single skill text with composer + query_latents.
    Returns (k, D_composer) tensor on CPU.
    """
    device = query_latents.device
    ids = tokenizer(skill_text, return_tensors="pt", truncation=True,
                    max_length=max_length, add_special_tokens=False
                    )["input_ids"].to(device)
    L = ids.shape[1]
    total_len = L + k

    embed = composer_inner.get_input_embeddings()
    text_emb = embed(ids)  # (1, L, D)
    queries = query_latents.to(text_emb.dtype).unsqueeze(0)  # (1, k, D)
    combined = torch.cat([text_emb, queries], dim=1)  # (1, L+k, D)
    attn_mask = torch.ones((1, total_len), dtype=torch.long, device=device)

    out = composer_inner.model(
        inputs_embeds=combined, attention_mask=attn_mask,
        use_cache=False, return_dict=True,
    )
    last_hidden = out.last_hidden_state  # (1, L+k, D)
    return last_hidden[0, -k:, :].cpu()  # (k, D_composer)


@torch.no_grad()
def build_latent_library(composer_inner, query_latents, latent_proj, k,
                         composer_tok, skills_json, device,
                         extra_meta=None):
    """Encode the 44 base skills IN-MEMORY (no ckpt I/O) and return a dict in the
    exact schema main() saves. Used by the trainer to emit a per-epoch lib.pt
    without writing a 15GB composer checkpoint. latent_proj may be nn.Identity.
    """
    all_skills = extract_individual_skills(skills_json)
    pad_id = composer_tok.pad_token_id or 0
    latent_library, skill_index = {}, {}
    for sk in all_skills:
        latent_composer = encode_one_skill(
            composer_inner, query_latents, k, sk["text"], composer_tok, pad_id,
        )  # (k, D_composer) on CPU
        # Match latent_proj's weight dtype (fp32 Linear vs bf16 Identity input)
        proj_in = latent_composer.to(device)
        if isinstance(latent_proj, nn.Linear):
            proj_in = proj_in.to(latent_proj.weight.dtype)
        latent_actor = latent_proj(proj_in).cpu()
        h = hash_skill(sk["text"])
        latent_library[h] = latent_actor.to(torch.bfloat16)
        skill_index[h] = {"id": sk["id"], "text": sk["text"][:100], "type": sk["type"]}
    D_composer = query_latents.shape[-1]
    D_actor = next(iter(latent_library.values())).shape[-1]
    out = {
        "latent_library": latent_library,
        "skill_index": skill_index,
        "query_latents": query_latents.detach().cpu(),
        "latents_per_skill": k,
        "hidden_size": D_actor,
        "variant": "TRAINED",
        "D_composer": D_composer,
        "D_actor": D_actor,
    }
    if extra_meta:
        out.update(extra_meta)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--composer-ckpt", required=True,
                        help="Path to Stage 1 Composer ckpt (TRAINED) or 'Qwen/Qwen2.5-3B-Instruct' (UNTRAINED)")
    parser.add_argument("--query-latents", default=None,
                        help="Path to query_latents_step{N}.pt from Stage 1 (TRAINED only). "
                             "If None (UNTRAINED), generates random ql + proj from --seed.")
    parser.add_argument("--actor-path", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Actor model path — needed to know D_actor for projection target")
    parser.add_argument("--skills-json", default=None,
                        help="Path to claude_style_skills.json. Defaults to SkillRL alfworld.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=["TRAINED", "UNTRAINED"], required=True)
    parser.add_argument("--latents-per-skill", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42,
                        help="For UNTRAINED: random init seed for ql + proj (matches Stage 1 init pattern)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skills_json = args.skills_json or str(
        SKILLRL_ROOT / "memory_data" / "alfworld" / "claude_style_skills.json"
    )

    # ----- Load Composer -----
    logger.info(f"Loading Composer from {args.composer_ckpt} (variant={args.variant})")
    composer_tok = AutoTokenizer.from_pretrained(args.composer_ckpt, trust_remote_code=True)
    if composer_tok.pad_token is None:
        composer_tok.pad_token = composer_tok.eos_token
    composer = AutoModelForCausalLM.from_pretrained(
        args.composer_ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(args.device)
    composer.eval()
    D_composer = composer.config.hidden_size
    logger.info(f"  D_composer = {D_composer}")

    # ----- Determine D_actor (for projection target) -----
    actor_cfg = AutoModelForCausalLM.from_pretrained(
        args.actor_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).config
    D_actor = actor_cfg.hidden_size
    logger.info(f"  D_actor    = {D_actor}")

    # ----- Load / construct query_latents + latent_proj -----
    if args.variant == "TRAINED":
        if args.query_latents is None:
            raise ValueError("--query-latents required for TRAINED variant")
        logger.info(f"Loading query_latents + latent_proj from {args.query_latents}")
        ckpt = torch.load(args.query_latents, map_location="cpu", weights_only=False)
        query_latents = ckpt["query_latents"].to(args.device).to(torch.bfloat16)  # (k, D_composer)
        if D_composer != D_actor:
            assert "latent_proj_state" in ckpt, "TRAINED ckpt must contain latent_proj"
            latent_proj = nn.Linear(D_composer, D_actor, bias=False).to(args.device, dtype=torch.bfloat16)
            latent_proj.load_state_dict({k: v.to(torch.bfloat16) for k, v in ckpt["latent_proj_state"].items()})
            latent_proj.eval()
        else:
            latent_proj = nn.Identity()
        logger.info(f"  TRAINED ql norm = {query_latents.float().norm().item():.4f}, k={query_latents.shape[0]}")
    else:  # UNTRAINED
        torch.manual_seed(args.seed)
        # Same init as Stage 1: randn * 0.02
        query_latents = (torch.randn(args.latents_per_skill, D_composer) * 0.02).to(
            args.device, dtype=torch.bfloat16,
        )
        if D_composer != D_actor:
            latent_proj = nn.Linear(D_composer, D_actor, bias=False).to(args.device, dtype=torch.bfloat16)
            with torch.no_grad():
                nn.init.normal_(latent_proj.weight, std=0.02)
            latent_proj.eval()
        else:
            latent_proj = nn.Identity()
        logger.info(f"  UNTRAINED ql norm = {query_latents.float().norm().item():.4f}, "
                    f"k={query_latents.shape[0]} (seed={args.seed})")

    # ----- Load skills -----
    all_skills = extract_individual_skills(skills_json)
    logger.info(f"Loaded {len(all_skills)} skills from {skills_json}")

    # ----- Encode -----
    pad_id = composer_tok.pad_token_id or 0
    latent_library = {}
    skill_index = {}
    for i, sk in enumerate(all_skills):
        latent_composer = encode_one_skill(
            composer, query_latents, args.latents_per_skill, sk["text"],
            composer_tok, pad_id,
        )  # (k, D_composer)
        # Project to actor space
        with torch.no_grad():
            latent_actor = latent_proj(latent_composer.to(args.device)).cpu()  # (k, D_actor)

        h = hash_skill(sk["text"])
        latent_library[h] = latent_actor.to(torch.bfloat16)
        skill_index[h] = {"id": sk["id"], "text": sk["text"][:100], "type": sk["type"]}

        if (i + 1) % 10 == 0 or i == len(all_skills) - 1:
            logger.info(f"  Encoded {i+1}/{len(all_skills)}: {sk['id']} ({sk['type']}) "
                        f"latent_actor.norm={latent_actor.float().norm().item():.4f}")

    out_path = output_dir / "latent_skill_library.pt"
    torch.save({
        "latent_library": latent_library,
        "skill_index": skill_index,
        "query_latents": query_latents.cpu(),
        "model_name": args.composer_ckpt,
        "latents_per_skill": args.latents_per_skill,
        "hidden_size": D_actor,  # Note: in Actor's hidden_size (post-projection)
        "seed": args.seed,
        "variant": args.variant,
        "D_composer": D_composer,
        "D_actor": D_actor,
    }, out_path)

    logger.info(f"Saved {len(latent_library)} skills to {out_path}")
    logger.info(f"  variant={args.variant}, k={args.latents_per_skill}, D_actor={D_actor}")


if __name__ == "__main__":
    main()
