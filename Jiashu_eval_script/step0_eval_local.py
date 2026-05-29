import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
import yaml

from agent_system.environments.env_manager import AlfWorldEnvironmentManager
from agent_system.environments.env_package.alfworld import (
    alfworld_projection,
    build_alfworld_envs,
)


TASKS = [
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Small-scale step-0 ALFWorld evaluation with a local HF model."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--env-num", type=int, default=8)
    parser.add_argument(
        "--game-offset",
        type=int,
        default=0,
        help="Starting index within the evaluation split.",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=None,
        help="Number of evaluation games to cover from game-offset onward.",
    )
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--eval-dataset",
        choices=["eval_in_distribution", "eval_out_of_distribution"],
        default="eval_in_distribution",
    )
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--history-length", type=int, default=10)
    parser.add_argument("--use-skills-only-memory", action="store_true")
    parser.add_argument(
        "--skills-json-path",
        default="/home/nvidia/Jiashu/SkillRL/memory_data/alfworld/claude_style_skills.json",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=["template", "embedding"],
        default="template",
    )
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--task-specific-top-k", type=int, default=None)
    parser.add_argument("--enable-dynamic-update", action="store_true")
    parser.add_argument("--update-threshold", type=float, default=0.4)
    parser.add_argument("--max-new-skills", type=int, default=3)
    parser.add_argument(
        "--embedding-model-path",
        default="Qwen/Qwen3-Embedding-0.6B",
    )
    parser.add_argument(
        "--save-dir",
        default="/home/nvidia/Jiashu/SkillRL/logs/step0_eval_local",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional HF attention backend such as sdpa/eager/flash_attention_2.",
    )
    parser.add_argument(
        "--merged-model-cache-dir",
        default=None,
        help=(
            "Optional directory for caching a merged HF checkpoint when "
            "--model-path points to a verl FSDP checkpoint. If unset, the "
            "merged model is stored inside the run directory as before."
        ),
    )
    return parser.parse_args()


def build_config(args):
    env_cfg = {
        "env_name": "alfworld/AlfredTWEnv",
        "seed": args.seed,
        "max_steps": args.max_steps,
        "history_length": args.history_length,
        "use_skills_only_memory": args.use_skills_only_memory,
        "use_retrieval_memory": False,
        "alfworld": {
            "eval_dataset": args.eval_dataset,
        },
    }

    if args.use_skills_only_memory:
        env_cfg["skills_only_memory"] = {
            "skills_json_path": args.skills_json_path,
            "retrieval_mode": args.retrieval_mode,
            "embedding_model_path": args.embedding_model_path,
            "top_k": args.top_k,
            "task_specific_top_k": args.task_specific_top_k,
            "enable_dynamic_update": args.enable_dynamic_update,
            "update_threshold": args.update_threshold,
            "max_new_skills": args.max_new_skills,
        }

    return OmegaConf.create(
        {
            "env": env_cfg
        }
    )


def build_env_manager(args, game_offset: int, num_games: int):
    base_alf_config_path = os.path.join(
        os.path.dirname(__file__),
        "../../agent_system/environments/env_package/alfworld/configs/config_tw.yaml",
    )
    with open(base_alf_config_path, "r", encoding="utf-8") as f:
        alf_cfg = yaml.safe_load(f)

    alf_cfg.setdefault("dataset", {})
    alf_cfg["dataset"]["eval_start_index"] = game_offset
    alf_cfg["dataset"]["num_eval_games"] = num_games

    temp_dir = tempfile.mkdtemp(prefix="alfworld_eval_cfg_", dir="/tmp")
    alf_config_path = os.path.join(temp_dir, "config_tw.yaml")
    with open(alf_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(alf_cfg, f, sort_keys=False)

    env_kwargs = {"eval_dataset": args.eval_dataset}
    resources_per_worker = {"num_cpus": 0.1, "num_gpus": 0.0}

    envs = build_alfworld_envs(
        alf_config_path=alf_config_path,
        seed=args.seed,
        env_num=1,
        group_n=1,
        resources_per_worker=resources_per_worker,
        is_train=False,
        env_kwargs=env_kwargs,
    )
    return AlfWorldEnvironmentManager(envs, alfworld_projection, build_config(args))


def resolve_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    return torch.float32


class LocalChatAgent:
    def __init__(self, args, run_dir: str):
        self.args = args
        self.model_path = self._resolve_model_path(args.model_path, run_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=args.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": args.trust_remote_code,
            "torch_dtype": resolve_torch_dtype(args.dtype),
            "device_map": "auto",
        }
        if args.attn_implementation:
            model_kwargs["attn_implementation"] = args.attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **model_kwargs,
        )
        self.model.eval()

    def _resolve_model_path(self, raw_model_path: str, run_dir: str) -> str:
        model_path = os.path.abspath(raw_model_path)
        actor_candidate = os.path.join(model_path, "actor")

        if self._is_hf_model_dir(model_path):
            return model_path

        if os.path.isdir(actor_candidate):
            if self._is_hf_model_dir(actor_candidate):
                return actor_candidate
            if self._is_verl_fsdp_actor_dir(actor_candidate):
                return self._merge_verl_fsdp_checkpoint(actor_candidate, run_dir)

        if self._is_verl_fsdp_actor_dir(model_path):
            return self._merge_verl_fsdp_checkpoint(model_path, run_dir)

        raise ValueError(
            f"Unsupported model path: {raw_model_path}. Expected a HF model dir, "
            f"a verl checkpoint root containing actor/, or a verl actor checkpoint dir."
        )

    @staticmethod
    def _is_hf_model_dir(path: str) -> bool:
        if not os.path.isdir(path):
            return False
        has_config = os.path.exists(os.path.join(path, "config.json"))
        has_weights = any(
            os.path.exists(os.path.join(path, name))
            for name in [
                "model.safetensors",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
                "model.safetensors.index.json",
            ]
        )
        return has_config and has_weights

    @staticmethod
    def _is_verl_fsdp_actor_dir(path: str) -> bool:
        if not os.path.isdir(path):
            return False
        has_config = os.path.exists(os.path.join(path, "config.json"))
        has_shards = any(
            name.startswith("model_world_size_") and name.endswith(".pt")
            for name in os.listdir(path)
        )
        return has_config and has_shards

    def _merge_verl_fsdp_checkpoint(self, actor_dir: str, run_dir: str) -> str:
        merged_dir = self.args.merged_model_cache_dir
        if merged_dir:
            merged_dir = os.path.abspath(merged_dir)
        else:
            merged_dir = os.path.join(run_dir, "merged_hf_model")

        if self._is_hf_model_dir(merged_dir):
            logging.info("Using cached merged HF model: %s", merged_dir)
            return merged_dir

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        merger_script = os.path.join(repo_root, "scripts", "model_merger.py")
        cmd = [
            sys.executable,
            merger_script,
            "merge",
            "--backend",
            "fsdp",
            "--local_dir",
            actor_dir,
            "--target_dir",
            merged_dir,
        ]
        logging.info("Merging verl FSDP checkpoint to HF format: %s", " ".join(cmd))
        subprocess.run(cmd, cwd=repo_root, check=True)

        if not self._is_hf_model_dir(merged_dir):
            raise RuntimeError(
                f"Checkpoint merge finished but merged HF model was not found at {merged_dir}"
            )
        return merged_dir

    def _format_input(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
        return self.tokenizer(prompt, return_tensors="pt").input_ids

    @torch.inference_mode()
    def get_response(self, prompt: str) -> str:
        input_ids = self._format_input(prompt).to(self.model.device)
        attention_mask = torch.ones_like(input_ids, device=input_ids.device)

        do_sample = self.args.temperature > 0
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.args.max_new_tokens,
            do_sample=do_sample,
            temperature=self.args.temperature if do_sample else None,
            top_p=self.args.top_p if do_sample else None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = outputs[0][input_ids.shape[-1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def extract_task_type(gamefile: str) -> str:
    for task in TASKS:
        if task in gamefile:
            return task
    return "other"


def extract_parsed_action(response_text: str) -> str:
    lower_text = response_text.lower()
    start_tag = "<action>"
    end_tag = "</action>"
    start_idx = lower_text.find(start_tag)
    end_idx = lower_text.find(end_tag)
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return ""
    return lower_text[start_idx + len(start_tag) : end_idx].strip()


def make_run_dir(args):
    os.makedirs(args.save_dir, exist_ok=True)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def setup_logging(run_dir: str):
    log_fp = os.path.join(run_dir, "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_fp, encoding="utf-8"), logging.StreamHandler()],
    )


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    success_by_task = defaultdict(lambda: {"success": 0, "total": 0})
    for item in episodes:
        task_type = item["task_type"]
        success_by_task[task_type]["total"] += 1
        success_by_task[task_type]["success"] += int(item["won"])

    overall = {
        "num_episodes": len(episodes),
        "num_success": sum(int(item["won"]) for item in episodes),
        "success_rate": (
            sum(int(item["won"]) for item in episodes) / len(episodes)
            if episodes
            else 0.0
        ),
        "by_task": {},
    }
    for task_type, stats in success_by_task.items():
        overall["by_task"][task_type] = {
            "success": stats["success"],
            "total": stats["total"],
            "success_rate": stats["success"] / stats["total"],
        }
    return overall


def resolve_total_games(args) -> int:
    if args.num_games is not None:
        return args.num_games
    return max(args.env_num, 1)


def main():
    args = parse_args()
    run_dir = make_run_dir(args)
    setup_logging(run_dir)
    total_games = resolve_total_games(args)

    logging.info("Run directory: %s", run_dir)
    logging.info(
        "Starting local step-0 evaluation | model_path=%s env_num=%d max_steps=%d eval_dataset=%s skills=%s retrieval_mode=%s",
        args.model_path,
        args.env_num,
        args.max_steps,
        args.eval_dataset,
        args.use_skills_only_memory,
        args.retrieval_mode,
    )
    logging.info(
        "Evaluation coverage | game_offset=%d num_games=%s",
        args.game_offset,
        str(total_games),
    )
    if args.env_num != 1:
        logging.warning(
            "--env-num=%d is ignored for execution concurrency. This script now runs "
            "strictly serial on a single GPU process. Total requested games=%d.",
            args.env_num,
            total_games,
        )

    agent = LocalChatAgent(args, run_dir)

    raw_responses_path = os.path.join(run_dir, "raw_responses.jsonl")
    episodes_path = os.path.join(run_dir, "episodes.jsonl")
    summary_path = os.path.join(run_dir, "summary.json")

    raw_rows: List[Dict[str, Any]] = []
    finished_episodes: List[Dict[str, Any]] = []

    try:
        start_time = time.time()

        episode_progress = tqdm(
            range(total_games),
            desc="ALFWorld eval episodes",
            file=sys.stdout,
            dynamic_ncols=True,
            leave=True,
        )

        for episode_idx in episode_progress:
            current_offset = args.game_offset + episode_idx
            episode_progress.set_postfix(offset=current_offset)
            logging.info(
                "Episode %d/%d | dataset index=%d",
                episode_idx + 1,
                total_games,
                current_offset,
            )

            env_manager = build_env_manager(args, game_offset=current_offset, num_games=1)
            obs, _infos = env_manager.reset(kwargs={})
            done_flag = False
            episode_trace: List[Dict[str, Any]] = []

            step_progress = tqdm(
                range(args.max_steps),
                desc=f"Game {current_offset}",
                file=sys.stdout,
                dynamic_ncols=True,
                leave=False,
            )

            for step_idx in step_progress:
                if done_flag:
                    break

                prompt = obs["text"][0]
                response_text = agent.get_response(prompt)
                parsed_action = extract_parsed_action(response_text)
                raw_rows.append(
                    {
                        "env_idx": episode_idx,
                        "dataset_idx": current_offset,
                        "step_idx": step_idx,
                        "prompt": prompt,
                        "response": response_text,
                    }
                )

                next_obs, rewards, dones, infos = env_manager.step([response_text])
                info = infos[0]
                step_record = {
                    "step_idx": step_idx,
                    "prompt": prompt,
                    "raw_response": response_text,
                    "parsed_action": parsed_action,
                    "reward": float(rewards[0]),
                    "done": bool(dones[0]),
                    "won": bool(info.get("won", False)),
                    "is_action_valid": bool(info.get("is_action_valid", 0)),
                    "gamefile": info.get("extra.gamefile", ""),
                    "next_observation": next_obs["anchor"][0],
                }
                episode_trace.append(step_record)
                done_flag = bool(dones[0])
                obs = next_obs
                step_progress.set_postfix(done=done_flag, won=bool(info.get("won", False)))

                if done_flag:
                    logging.info(
                        "Episode %d finished at step %d | won=%s",
                        current_offset,
                        step_idx,
                        bool(info.get("won", False)),
                    )
                    gamefile = info.get("extra.gamefile", "")
                    finished_episodes.append(
                        {
                            "env_idx": episode_idx,
                            "dataset_idx": current_offset,
                            "task": env_manager.tasks[0],
                            "task_type": extract_task_type(gamefile),
                            "won": bool(info.get("won", False)),
                            "num_steps": len(episode_trace),
                            "gamefile": gamefile,
                            "trajectory": episode_trace,
                        }
                    )
                    break

            step_progress.close()

            if not done_flag:
                finished_episodes.append(
                    {
                        "env_idx": episode_idx,
                        "dataset_idx": current_offset,
                        "task": env_manager.tasks[0],
                        "task_type": "unfinished",
                        "won": False,
                        "num_steps": len(episode_trace),
                        "gamefile": "",
                        "trajectory": episode_trace,
                    }
                )

            if hasattr(env_manager, "envs") and hasattr(env_manager.envs, "close"):
                env_manager.envs.close()

        summary = summarize(finished_episodes)
        summary["elapsed_seconds"] = time.time() - start_time
        summary["model_path"] = args.model_path
        summary["eval_dataset"] = args.eval_dataset
        summary["game_offset"] = args.game_offset
        summary["num_games"] = total_games
        summary["execution_mode"] = "serial_single_env"
        summary["requested_env_num"] = args.env_num
        summary["use_skills_only_memory"] = args.use_skills_only_memory
        if args.use_skills_only_memory:
            summary["skills_only_memory"] = {
                "skills_json_path": args.skills_json_path,
                "retrieval_mode": args.retrieval_mode,
                "top_k": args.top_k,
                "task_specific_top_k": args.task_specific_top_k,
                "enable_dynamic_update": args.enable_dynamic_update,
                "update_threshold": args.update_threshold,
                "max_new_skills": args.max_new_skills,
                "embedding_model_path": args.embedding_model_path,
            }

        write_jsonl(raw_responses_path, raw_rows)
        write_jsonl(episodes_path, finished_episodes)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logging.info("Finished. Summary saved to %s", summary_path)
        logging.info("Overall success rate: %.4f", summary["success_rate"])
        for task_type, stats in summary["by_task"].items():
            logging.info(
                "%s | %.4f (%d/%d)",
                task_type,
                stats["success_rate"],
                stats["success"],
                stats["total"],
            )

    finally:
        env_manager.close()


if __name__ == "__main__":
    main()
