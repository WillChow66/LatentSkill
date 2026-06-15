"""Debug the same7b k=1 anomaly (0/105 vs k=2's 45.7%).

Runs N episodes for both k=1 and k=2 same7b variants and prints the model's
ACTUAL generations step-by-step, so we can see whether k=1 output is gibberish,
looping, format-broken, or coherent-but-wrong.

Reuses the eval pipeline's own helpers (no logic duplication).
"""

import argparse
import logging

import torch

from src.eval_latent_alfworld import (
    build_skill_hash_lookup, make_env_config, generate_with_latent,
    SKILLRL_ROOT,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import partial

logger = logging.getLogger(__name__)


def run_variant(tag, model_path, lib_path, episodes, steps, device="cuda"):
    from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
    from agent_system.environments.env_manager import AlfWorldEnvironmentManager

    print(f"\n{'='*80}\n### VARIANT {tag}: {model_path}\n{'='*80}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.eval()

    ckpt = torch.load(lib_path, map_location="cpu", weights_only=False)
    latent_library = ckpt["latent_library"]
    print(f"lib: {len(latent_library)} skills, k={ckpt['latents_per_skill']}")

    skills_json_path = str(SKILLRL_ROOT / "memory_data" / "alfworld" / "claude_style_skills.json")
    skill_hashes = build_skill_hash_lookup(skills_json_path)

    config = make_env_config(history_length=10, top_k=6)
    alf_config_path = str(SKILLRL_ROOT / "agent_system" / "environments" / "env_package" / "alfworld" / "configs" / "config_tw.yaml")
    raw_envs = build_alfworld_envs(
        alf_config_path, seed=42, env_num=1, group_n=1, is_train=False,
        env_kwargs={"eval_dataset": "eval_in_distribution"},
        resources_per_worker={"num_cpus": 1},
    )
    env_manager = AlfWorldEnvironmentManager(raw_envs, partial(alfworld_projection), config)

    for ep in range(episodes):
        observations, infos = env_manager.reset(kwargs={})
        print(f"\n--- {tag} EPISODE {ep} ---")
        for step in range(steps):
            instruction = observations["text"][0]
            task_desc = env_manager.tasks[0] if hasattr(env_manager, "tasks") else ""
            if step == 0:
                # Show the tail of the instruction (task statement region)
                print(f"[instruction tail 400 chars]: ...{instruction[-400:]!r}")
            response, in_tok, lat_tok, out_tok, elapsed = generate_with_latent(
                instruction, model, tokenizer, latent_library, skill_hashes,
                task_desc, device, do_sample=True, temperature=0.4, top_p=1.0,
                max_new_tokens=512,
            )
            print(f"[{tag} ep{ep} step{step}] latents={lat_tok} out_tok={out_tok}")
            print(f"  RESPONSE: {response[:600]!r}")
            observations, rewards, dones, step_infos = env_manager.step([response])
            if dones[0]:
                won = step_infos[0].get("won", False) if step_infos[0] else False
                print(f"  EPISODE DONE at step {step}, won={won}")
                break

    raw_envs.close()
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    base = "/work/nvme/bfdz/xzhou10/checkpoints"
    run_variant(
        "k1-BROKEN",
        f"{base}/latent_lib_same7b_k1/actor_expanded_vocab",
        f"{base}/latent_lib_same7b_k1/latent_skill_library.pt",
        args.episodes, args.steps,
    )
    run_variant(
        "k2-WORKS",
        f"{base}/latent_lib_same7b_k2/actor_expanded_vocab",
        f"{base}/latent_lib_same7b_k2/latent_skill_library.pt",
        args.episodes, args.steps,
    )


if __name__ == "__main__":
    main()
