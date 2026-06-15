"""
ALFWorld Evaluation for Latent Skills SFT Checkpoint

Loads our full-param SFT checkpoint and pre-computed latent skill library.
For each step: splits instruction at skills boundary, looks up pre-computed
latent tokens, does in-place replacement, generates with inputs_embeds.

Usage:
  python -m src.eval_latent_alfworld \
      --model /path/to/sft_checkpoint \
      --latent-library data/latent_library/latent_skill_library.pt \
      --num-episodes 140
"""

import sys
import os
import json
import time
import hashlib
import re
import argparse
import logging
from pathlib import Path
from functools import partial
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

SKILLRL_ROOT = Path(__file__).parent.parent / "SkillRL"
sys.path.insert(0, str(SKILLRL_ROOT))
sys.path.insert(0, str(SKILLRL_ROOT / "agent_system" / "environments" / "env_package" / "alfworld"))

from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
from agent_system.environments.env_manager import AlfWorldEnvironmentManager

# Skills section markers (same as training)
SKILLS_START_MARKER = "## Retrieved Relevant Experience"
SKILLS_END_MARKER = "## Current Progress"

TASK_TYPES = [
    "pick_and_place", "pick_two_obj_and_place", "look_at_obj_in_light",
    "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]


def hash_skill(text):
    return hashlib.md5(text.encode()).hexdigest()


def split_instruction(text):
    """Split instruction at skills boundary. Returns (before, skills_section, after)."""
    start = text.find(SKILLS_START_MARKER)
    end = text.find(SKILLS_END_MARKER)
    if start == -1 or end == -1:
        return text, "", ""
    return text[:start], text[start:end], text[end:]


def build_skill_hash_lookup(skills_json_path):
    """Build task_type -> list of skill hashes, using SAME format as encode_skill_library.py."""
    with open(skills_json_path) as f:
        data = json.load(f)

    all_hashes = {}  # task_type -> [hash1, hash2, ...]

    # General skills (used for all tasks)
    general_hashes = []
    for s in data.get("general_skills", []):
        text = f"**{s['title']}**: {s['principle']}"
        if "when_to_apply" in s:
            text += f" Apply when: {s['when_to_apply']}"
        general_hashes.append(hash_skill(text))
    all_hashes["general"] = general_hashes

    # Task-specific skills
    for task_type, skills in data.get("task_specific_skills", {}).items():
        task_hashes = []
        for s in skills:
            text = f"**{s['title']}**: {s['principle']}"
            if "when_to_apply" in s:
                text += f" Apply when: {s['when_to_apply']}"
            task_hashes.append(hash_skill(text))
        all_hashes[task_type] = task_hashes

    # Common mistakes
    mistake_hashes = []
    for i, m in enumerate(data.get("common_mistakes", [])):
        if "title" in m and "principle" in m:
            text = f"**{m['title']}**: {m['principle']}"
            mistake_hashes.append(hash_skill(text))
    all_hashes["mistakes"] = mistake_hashes

    return all_hashes


# Task type detection from instruction (same keywords as SkillsOnlyMemory)
def detect_skill_task_type(task_description):
    goal = task_description.lower()
    if "look at" in goal and "under" in goal:
        return "look_at_obj_in_light"
    elif "clean" in goal:
        return "clean"
    elif "heat" in goal:
        return "heat"
    elif "cool" in goal:
        return "cool"
    elif "examine" in goal or "find" in goal:
        return "examine"
    else:
        return "pick_and_place"


def detect_task_type(gamefile):
    for t in TASK_TYPES:
        if t in gamefile:
            return t
    return "unknown"


def make_env_config(history_length: int = 10, top_k: int = 6):
    skills_path = str(SKILLRL_ROOT / "memory_data" / "alfworld" / "claude_style_skills.json")
    return OmegaConf.create({
        "env": {
            "env_name": "alfworld/AlfredTWEnv",
            "history_length": history_length,
            "max_steps": 50,
            "seed": 42,
            "alfworld": {"eval_dataset": "eval_in_distribution"},
            "use_skills_only_memory": True,
            "skills_only_memory": {
                "skills_json_path": skills_path,
                "retrieval_mode": "template",
                "embedding_model_path": None,
                "task_specific_top_k": None,
                "top_k": top_k,
            },
        },
        "data": {"train_batch_size": 1, "val_batch_size": 1},
    })


@torch.no_grad()
def skill_token_ids(tokenizer):
    """IDs of SKILL_* special tokens (present only on vocab-expanded actors).
    Their lm_head rows are zeroed at bake time (logit pinned to 0 ≈ never wins
    vs typical 10-25 top logits), but explicit suppression is free insurance —
    especially for RL-trained actors whose lm_head rows may drift from zero.
    Returns [] on vanilla models → no-op."""
    return [tid for tok, tid in tokenizer.get_added_vocab().items()
            if tok.startswith("SKILL_")]


def generate_with_latent(instruction, model, tokenizer, latent_library, skill_hashes,
                         task_description, device, do_sample: bool = True,
                         temperature: float = 0.4, top_p: float = 1.0,
                         max_new_tokens: int = 512, suppress_ids=None):
    """Generate action with latent skill in-place replacement."""
    start_time = time.time()

    # Apply chat template (same as training)
    messages = [{"role": "user", "content": instruction}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Split at skills boundary
    before, skills_text, after = split_instruction(chat_text)

    # Look up pre-computed latent tokens by task type (same hashes as encoding)
    task_type = detect_skill_task_type(task_description)
    relevant_hashes = skill_hashes.get("general", []) + skill_hashes.get(task_type, []) + skill_hashes.get("mistakes", [])

    latent_list = []
    for h in relevant_hashes:
        if h in latent_library:
            latent_list.append(latent_library[h].to(device))

    # Get embeddings for before/after (explicit .long() to avoid dtype issues)
    before_ids = tokenizer.encode(before, add_special_tokens=False, return_tensors="pt").long().to(device)
    after_ids = tokenizer.encode(after, add_special_tokens=False, return_tensors="pt").long().to(device)

    embed = model.get_input_embeddings()
    before_emb = embed(before_ids)  # (1, before_len, D)
    after_emb = embed(after_ids)    # (1, after_len, D)

    # In-place: [before | latent_tokens | after]
    if latent_list:
        latent_tokens = torch.cat(latent_list, dim=0).unsqueeze(0).to(before_emb.dtype)  # (1, N*k, D)
        full_emb = torch.cat([before_emb, latent_tokens, after_emb], dim=1)
        latent_count = latent_tokens.shape[1]
    else:
        full_emb = torch.cat([before_emb, after_emb], dim=1)
        latent_count = 0

    attn_mask = torch.ones(full_emb.shape[:2], dtype=torch.long, device=device)
    input_token_count = before_ids.shape[1] + after_ids.shape[1]

    outputs = model.generate(
        inputs_embeds=full_emb,
        attention_mask=attn_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        suppress_tokens=suppress_ids if suppress_ids else None,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_token_count = len(outputs[0])

    elapsed = time.time() - start_time
    return response, input_token_count, latent_count, output_token_count, elapsed


def run_evaluation(
    model_path: str,
    latent_library_path: str,
    num_episodes: int = 140,
    max_steps: int = 50,
    device: str = "cuda",
    seed: int = 42,
    output_path: str = None,
    do_sample: bool = True,
    temperature: float = 0.4,
    top_p: float = 1.0,
    max_new_tokens: int = 512,
    history_length: int = 10,
    top_k: int = 6,
):
    logger.info(f"=== Latent Skills Evaluation: {num_episodes} episodes ===")
    logger.info(f"Sampling: do_sample={do_sample} temperature={temperature} top_p={top_p}")
    logger.info(f"Env: history_length={history_length} top_k={top_k}")

    # Load model
    logger.info(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    suppress_ids = skill_token_ids(tokenizer)
    if suppress_ids:
        logger.info(f"Suppressing {len(suppress_ids)} SKILL_* tokens at generation "
                    f"(vocab-expanded actor detected)")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.eval()

    # Load pre-computed latent skill library
    logger.info(f"Loading latent library: {latent_library_path}")
    ckpt = torch.load(latent_library_path, map_location="cpu")
    latent_library = ckpt["latent_library"]
    logger.info(f"  {len(latent_library)} skills, k={ckpt['latents_per_skill']}")

    # Build skill hash lookup (same format as encode_skill_library.py)
    skills_json_path = str(SKILLRL_ROOT / "memory_data" / "alfworld" / "claude_style_skills.json")
    skill_hashes = build_skill_hash_lookup(skills_json_path)
    total_hashes = sum(len(v) for v in skill_hashes.values())
    logger.info(f"  Skill hash lookup: {total_hashes} hashes across {len(skill_hashes)} categories")

    # Build ALFWorld environment (with skills in prompt — we'll replace them with latent)
    config = make_env_config(history_length=history_length, top_k=top_k)
    alf_config_path = str(SKILLRL_ROOT / "agent_system" / "environments" / "env_package" / "alfworld" / "configs" / "config_tw.yaml")

    logger.info("Building ALFWorld environment...")
    raw_envs = build_alfworld_envs(
        alf_config_path, seed, env_num=1, group_n=1, is_train=False,
        env_kwargs={"eval_dataset": "eval_in_distribution"},
        resources_per_worker={"num_cpus": 1},
    )
    env_manager = AlfWorldEnvironmentManager(raw_envs, partial(alfworld_projection), config)

    # Evaluation loop
    results_by_type = defaultdict(lambda: {"success": 0, "total": 0})
    all_episodes = []

    for ep in range(num_episodes):
        try:
            observations, infos = env_manager.reset(kwargs={})
            done = False
            won = False
            ep_input_tokens = 0
            ep_latent_tokens = 0
            ep_output_tokens = 0
            ep_time = 0.0
            num_steps = 0

            for step in range(max_steps):
                if done:
                    break

                instruction = observations['text'][0]
                task_desc = env_manager.tasks[0] if hasattr(env_manager, 'tasks') else ""
                response, in_tok, lat_tok, out_tok, elapsed = generate_with_latent(
                    instruction, model, tokenizer, latent_library, skill_hashes, task_desc, device,
                    do_sample=do_sample, temperature=temperature, top_p=top_p,
                    max_new_tokens=max_new_tokens, suppress_ids=suppress_ids,
                )

                ep_input_tokens += in_tok
                ep_latent_tokens += lat_tok
                ep_output_tokens += out_tok
                ep_time += elapsed
                num_steps += 1

                observations, rewards, dones, step_infos = env_manager.step([response])
                done = dones[0]
                won = step_infos[0].get("won", False) if step_infos[0] else False

            gamefile = (infos[0] if infos else {}).get("extra.gamefile", "")
            task_type = detect_task_type(gamefile)
            results_by_type[task_type]["total"] += 1
            if won:
                results_by_type[task_type]["success"] += 1

            ep_data = {
                "episode": ep, "task_type": task_type, "success": won,
                "steps": num_steps,
                "input_tokens": ep_input_tokens,
                "latent_tokens": ep_latent_tokens,
                "output_tokens": ep_output_tokens,
                "total_tokens": ep_input_tokens + ep_latent_tokens + ep_output_tokens,
                "time": ep_time,
            }
            all_episodes.append(ep_data)

            status = "SUCCESS" if won else "FAIL"
            logger.info(
                f"Ep {ep+1}/{num_episodes} [{task_type}]: {status} | "
                f"Steps: {num_steps} | "
                f"Tokens: {ep_input_tokens}+{ep_latent_tokens}L+{ep_output_tokens} | "
                f"Time: {ep_time:.1f}s"
            )

        except Exception as e:
            logger.warning(f"Episode {ep+1} error: {e}", exc_info=True)
            continue

    # Summary
    total = sum(v["total"] for v in results_by_type.values())
    total_success = sum(v["success"] for v in results_by_type.values())
    overall_rate = total_success / total if total > 0 else 0

    logger.info(f"\n{'='*60}")
    logger.info(f"Overall success: {overall_rate:.1%} ({total_success}/{total})")
    for tt in TASK_TYPES:
        counts = results_by_type.get(tt, {"success": 0, "total": 0})
        if counts["total"] > 0:
            rate = counts["success"] / counts["total"]
            logger.info(f"  {tt}: {rate:.1%} ({counts['success']}/{counts['total']})")

    avg_tokens = sum(e["total_tokens"] for e in all_episodes) / max(len(all_episodes), 1)
    avg_time = sum(e["time"] for e in all_episodes) / max(len(all_episodes), 1)
    logger.info(f"Avg tokens/episode: {avg_tokens:.0f}")
    logger.info(f"Avg time/episode: {avg_time:.1f}s")
    logger.info(f"{'='*60}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "mode": "latent_skills",
                "model": model_path,
                "overall_success_rate": overall_rate,
                "by_task_type": {k: dict(v) for k, v in results_by_type.items()},
                "avg_tokens": avg_tokens,
                "avg_time": avg_time,
                "episodes": all_episodes,
            }, f, indent=2)
        logger.info(f"Saved to {output_path}")

    raw_envs.close()
    return overall_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to SFT checkpoint")
    parser.add_argument("--latent-library", type=str, required=True, help="Path to latent_skill_library.pt")
    parser.add_argument("--num-episodes", type=int, default=140)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--do-sample", action="store_true", default=True,
                        help="Use sampling (default True; pass --no-do-sample for greedy)")
    parser.add_argument("--no-do-sample", dest="do_sample", action="store_false")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--history-length", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_evaluation(
        model_path=args.model,
        latent_library_path=args.latent_library,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        output_path=args.output,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        history_length=args.history_length,
        top_k=args.top_k,
    )
