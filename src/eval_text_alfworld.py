"""ALFWorld evaluation with text skills (no latent replacement).

Used for:
  - Baseline (Qwen2.5-7B-Instruct, no fine-tune)
  - SkillRL RL ckpt (Jianwen/Alfworld-7B-RL after FSDP→HF merge)

Usage:
  python -m src.eval_text_alfworld \
      --model /path/to/hf_ckpt \
      --output results/baseline.json \
      --num-episodes 140
"""

import sys
import os
import json
import time
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

TASK_TYPES = [
    "pick_and_place", "pick_two_obj_and_place", "look_at_obj_in_light",
    "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]


def detect_task_type(gamefile):
    for t in TASK_TYPES:
        if t in gamefile:
            return t
    return "unknown"


def make_env_config(top_k: int = 6, history_length: int = 10, use_skills: bool = True):
    skills_path = str(SKILLRL_ROOT / "memory_data" / "alfworld" / "claude_style_skills.json")
    return OmegaConf.create({
        "env": {
            "env_name": "alfworld/AlfredTWEnv",
            "history_length": history_length,
            "max_steps": 50,
            "seed": 42,
            "alfworld": {"eval_dataset": "eval_in_distribution"},
            "use_skills_only_memory": use_skills,
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
def generate_text(instruction, model, tokenizer, device, do_sample: bool = True,
                  temperature: float = 0.4, top_p: float = 1.0,
                  max_new_tokens: int = 512):
    """Standard text generation with chat template."""
    start_time = time.time()

    messages = [{"role": "user", "content": instruction}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    input_ids = tokenizer.encode(chat_text, add_special_tokens=False, return_tensors="pt").to(device)
    attn_mask = torch.ones_like(input_ids)
    input_token_count = input_ids.shape[1]

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    response_ids = outputs[0, input_ids.shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    output_token_count = response_ids.shape[0]

    elapsed = time.time() - start_time
    return response, input_token_count, output_token_count, elapsed


def run_evaluation(model_path: str, num_episodes: int = 140, max_steps: int = 50,
                   device: str = "cuda", seed: int = 42, output_path: str = None,
                   top_k: int = 6, do_sample: bool = True, temperature: float = 0.4,
                   top_p: float = 1.0, max_new_tokens: int = 512,
                   history_length: int = 10, use_skills: bool = True):
    mode_str = "Text Skills" if use_skills else "NO-SKILL baseline"
    logger.info(f"=== {mode_str} Evaluation: {num_episodes} episodes ===")
    logger.info(f"Model: {model_path}")
    logger.info(f"Sampling: do_sample={do_sample} temperature={temperature} top_p={top_p}")
    logger.info(f"Env: history_length={history_length} top_k={top_k} use_skills={use_skills}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.eval()

    config = make_env_config(top_k=top_k, history_length=history_length, use_skills=use_skills)
    alf_config_path = str(SKILLRL_ROOT / "agent_system" / "environments" / "env_package" / "alfworld" / "configs" / "config_tw.yaml")

    logger.info("Building ALFWorld environment...")
    raw_envs = build_alfworld_envs(
        alf_config_path, seed, env_num=1, group_n=1, is_train=False,
        env_kwargs={"eval_dataset": "eval_in_distribution"},
        resources_per_worker={"num_cpus": 1},
    )
    env_manager = AlfWorldEnvironmentManager(raw_envs, partial(alfworld_projection), config)

    results_by_type = defaultdict(lambda: {"success": 0, "total": 0})
    all_episodes = []

    for ep in range(num_episodes):
        try:
            observations, infos = env_manager.reset(kwargs={})
            done = False
            won = False
            ep_input = 0
            ep_output = 0
            ep_time = 0.0
            num_steps = 0

            for step in range(max_steps):
                if done:
                    break

                instruction = observations['text'][0]
                response, in_tok, out_tok, elapsed = generate_text(
                    instruction, model, tokenizer, device,
                    do_sample=do_sample, temperature=temperature, top_p=top_p,
                    max_new_tokens=max_new_tokens,
                )

                ep_input += in_tok
                ep_output += out_tok
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

            all_episodes.append({
                "episode": ep, "task_type": task_type, "success": won,
                "steps": num_steps,
                "input_tokens": ep_input,
                "output_tokens": ep_output,
                "total_tokens": ep_input + ep_output,
                "time": ep_time,
            })

            status = "SUCCESS" if won else "FAIL"
            logger.info(
                f"Ep {ep+1}/{num_episodes} [{task_type}]: {status} | "
                f"Steps: {num_steps} | Tokens: {ep_input}+{ep_output} | Time: {ep_time:.1f}s"
            )

        except Exception as e:
            logger.warning(f"Episode {ep+1} error: {e}", exc_info=True)
            continue

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
                "mode": "text_skills" if use_skills else "baseline",
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
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num-episodes", type=int, default=140)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--do-sample", action="store_true", default=True,
                        help="Use sampling (default True; pass --no-do-sample for greedy)")
    parser.add_argument("--no-do-sample", dest="do_sample", action="store_false")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--history-length", type=int, default=10)
    parser.add_argument("--no-skills", action="store_true",
                        help="Run TRUE no-skill baseline (no text skills in prompt). "
                             "Default: text skills included (use_skills_only_memory=True).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_evaluation(
        model_path=args.model,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        top_k=args.top_k,
        output_path=args.output,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        history_length=args.history_length,
        use_skills=not args.no_skills,
    )
