"""Stage-3 RL (GRPO) on Modal — k=8 latent-skill actor, X2 DISABLED (v1).

EXPECT MULTI-ROUND DEBUGGING: verl + vLLM + ray + alfworld is a heavy stack;
the image will likely need several rebuild/fix iterations (version pins, flash-attn,
ray init, alfworld game-data paths). This file is the scaffold; smoke first.

Design (v1, static latent RL — simplest correct thing):
  - Actor   = /vol/rl_assets/actor_k8_expanded_vocab  (baked: 352 SKILL rows, untied LM head)
  - Skills  = static 44, latent_token_mode=True, latents_per_skill=8 (the k-hardcode fix)
  - X2 dynamic skill bank = OFF (enable_dynamic_update=False) → composer NOT needed alive,
    no OpenAI key needed. Matches SkillRL "w/o Dynamic Evolution" ablation (84.4%).
  - verl GRPO, SkillRL hyperparams; vLLM TP=4 on H200×4 (113GB/GPU need).
Game data + ckpts come from the Volume (mounted at /vol). Code (SkillRL/verl + src)
is baked into the image from .modal_stage_rl.

Smoke:  modal run modal_rl.py --epochs 2 --train-size 8 --val-size 8 --test-freq 1
Full:   modal run modal_rl.py            (150 epochs, SkillRL defaults)
"""
import subprocess
import modal

app = modal.App("latentskill-rl")

# vLLM official image = CUDA + torch + vllm 0.8.4 + flash-attn preinstalled (x86).
image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.8.4", add_python="3.11")
    .pip_install(
        "transformers==4.51.1",
        "tensordict<=0.6.2",
        "ray[default]",
        "codetiming", "dill", "hydra-core", "liger-kernel", "pylatexenc",
        "torchdata", "wandb", "peft", "omegaconf",
        "datasets", "pandas", "pyarrow>=19.0.0",
        "textworld==1.7.0", "fast-downward-textworld", "gymnasium==0.29.1",
        "sentence-transformers", "faiss-cpu", "networkx", "h5py",
        "opencv-python-headless", "pycocotools",
    )
    # install our verl fork editable (has the SkillRL/X2 changes)
    .run_commands("echo 'verl fork installed from /root/app/SkillRL at runtime via PYTHONPATH'")
    .add_local_dir(".modal_stage_rl", "/root/app")
)

vol = modal.Volume.from_name("latentskill", create_if_missing=True)

RL_ENV = {
    "HF_HOME": "/vol/hf_cache",
    "HF_HUB_CACHE": "/vol/hf_cache/hub",
    "ALFWORLD_DATA": "/vol/alfworld_data",
    "PYTHONPATH": "/root/app:/root/app/SkillRL:"
                  "/root/app/SkillRL/agent_system/environments/env_package/alfworld",
    "PYTHONUNBUFFERED": "1",
    "HYDRA_FULL_ERROR": "1",
    "RAY_memory_monitor_refresh_ms": "0",
    "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
    "TOKENIZERS_PARALLELISM": "false",
}


@app.function(image=image, volumes={"/vol": vol}, timeout=1800)
def import_check():
    """CPU-only smoke of the image + imports before burning H200. Catches the
    usual version/missing-pkg failures cheaply."""
    import os, subprocess
    os.environ.update(RL_ENV)
    subprocess.run(["pip", "install", "-e", ".", "--no-deps"],
                   cwd="/root/app/SkillRL", check=False)
    checks = [
        "import verl; print('verl', verl.__file__)",
        "import vllm; print('vllm', vllm.__version__)",
        "import transformers; print('transformers', transformers.__version__)",
        "import ray, tensordict, textworld; print('ray/tensordict/textworld ok')",
        "from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection; print('alfworld env import ok')",
        "from agent_system.memory.skills_only_memory import SkillsOnlyMemory; "
        "m=SkillsOnlyMemory('memory_data/alfworld/claude_style_skills.json', latent_token_mode=True, latents_per_skill=8); print('SkillsOnlyMemory k=8 ok')",
        "import os; print('actor exists:', os.path.exists('/vol/rl_assets/actor_k8_expanded_vocab/config.json'))",
        "import os; print('alfworld train exists:', os.path.exists('/vol/alfworld_data/json_2.1.1/train'))",
    ]
    for c in checks:
        print(f"--- {c[:60]}")
        r = subprocess.run(["python3", "-c", c], cwd="/root/app/SkillRL")
        if r.returncode != 0:
            return f"IMPORT CHECK FAILED at: {c[:80]}"
    return "ALL IMPORT CHECKS PASSED — safe to run train_rl on H200"


@app.function(image=image, gpu="H200:4", cpu=32, volumes={"/vol": vol},
              timeout=24 * 3600, secrets=[])
def train_rl(epochs: int = 150, train_size: int = 16, val_size: int = 64,
             group_size: int = 8, test_freq: int = 5):
    import os
    os.environ.update(RL_ENV)
    # editable-install verl fork so `import verl` resolves to our patched copy
    subprocess.run(["pip", "install", "-e", ".", "--no-deps"],
                   cwd="/root/app/SkillRL", check=False)

    actor = "/vol/rl_assets/actor_k8_expanded_vocab"
    out = "/vol/rl_assets/rl_k8_static_out"
    os.makedirs(out, exist_ok=True)
    assert os.path.exists(actor), f"baked actor missing: {actor}"

    cmd = [
        "python3", "-m", "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        "data.train_files=/vol/rl_assets/rl_parquet/train.parquet",
        "data.val_files=/vol/rl_assets/rl_parquet/test.parquet",
        f"data.train_batch_size={train_size}",
        f"data.val_batch_size={val_size}",
        "data.max_prompt_length=4096",
        "data.max_response_length=512",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "data.return_raw_chat=True",
        f"actor_rollout_ref.model.path={actor}",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.actor.ppo_mini_batch_size=128",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.01",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=4",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.5",
        "actor_rollout_ref.rollout.enable_chunked_prefill=True",
        "actor_rollout_ref.rollout.enforce_eager=False",
        "actor_rollout_ref.rollout.free_cache_engine=False",
        "actor_rollout_ref.rollout.max_num_batched_tokens=8192",
        "actor_rollout_ref.rollout.max_num_seqs=512",
        "actor_rollout_ref.rollout.val_kwargs.temperature=0.4",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.use_invalid_action_penalty=True",
        "actor_rollout_ref.actor.invalid_action_penalty_coef=0.1",
        "algorithm.use_kl_in_reward=False",
        "env.env_name=alfworld/AlfredTWEnv",
        "env.seed=0", "env.max_steps=50",
        f"env.rollout.n={group_size}",
        "env.resources_per_worker.num_cpus=0.1",
        "+env.use_skills_only_memory=True",
        "+env.skills_only_memory.skills_json_path=memory_data/alfworld/claude_style_skills.json",
        "+env.skills_only_memory.top_k=6",
        "+env.skills_only_memory.latent_token_mode=True",
        "+env.skills_only_memory.latents_per_skill=8",   # the k-hardcode fix
        "+env.skills_only_memory.enable_dynamic_update=False",  # X2 OFF in v1
        "trainer.critic_warmup=0",
        "trainer.logger=[console,wandb]",
        "trainer.project_name=latentskill",
        "trainer.experiment_name=rl_k8_static_modal",
        "trainer.n_gpus_per_node=4",
        "trainer.nnodes=1",
        f"trainer.default_local_dir={out}",
        "trainer.save_freq=10",
        f"trainer.test_freq={test_freq}",
        f"trainer.total_epochs={epochs}",
        "trainer.val_before_train=True",
        "trainer.max_actor_ckpt_to_keep=2",
    ]
    print("=== RL:", " ".join(cmd))
    subprocess.run(cmd, cwd="/root/app/SkillRL", check=True)
    vol.commit()
    return f"RL done (epochs={epochs}) -> {out}"


@app.local_entrypoint()
def main(epochs: int = 150, train_size: int = 16, val_size: int = 64,
         group_size: int = 8, test_freq: int = 5, check: bool = False):
    if check:
        print(import_check.remote())
    else:
        print(train_rl.remote(epochs, train_size, val_size, group_size, test_freq))
