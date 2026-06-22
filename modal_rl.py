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
# NOTE (Jun 15 fix): the image ships an ENTRYPOINT (vllm api_server) → Modal
# crash-loops on container start unless cleared with .entrypoint([]). And do NOT
# add_python: vllm lives in the image's native python; a fresh add_python 3.11
# would not have vllm → `import vllm` fails. Use the image's python directly.
image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.8.4")
    .entrypoint([])
    # image ships `python3` (with vllm) but not `python`; Modal's pip_install /
    # runner call `python` → symlink it before any pip step. Keeps the image's
    # vllm-bearing interpreter as the one we install into and run under.
    .run_commands("ln -sf $(command -v python3) /usr/local/bin/python")
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
    # flash_attn: vllm image ships only vLLM's internal kernels, NOT the standalone
    # `flash_attn` pkg that verl/HF need for attn_implementation=flash_attention_2 +
    # use_remove_padding (varlen). Prebuilt wheel EXACTLY matched to the image probe:
    # py3.12 / torch2.6 / cu12 / cxx11abi=False (wrong ABI = import-ok but segfault).
    .pip_install(
        "https://github.com/Dao-AILab/flash-attention/releases/download/"
        "v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-"
        "cp312-cp312-linux_x86_64.whl",
        "einops",
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
    # wandb ONLINE: live remote curves. Requires Modal secret `wandb-secret`
    # (WANDB_API_KEY=...) wired into train_rl. Entity medagent (personal disabled).
    # If the key is ever unavailable, set WANDB_MODE=offline (no key needed).
    "WANDB_MODE": "online",
    "WANDB_ENTITY": "medagent",
    "WANDB_DIR": "/vol/rl_assets/wandb",
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


@app.function(image=image, volumes={"/vol": vol}, timeout=600)
def diag():
    """Cheap CPU probe of the image's torch/cuda/abi so we pick the EXACT
    matching flash-attn prebuilt wheel (wrong ABI = import-ok but runtime segfault)."""
    import subprocess
    subprocess.run(["python3", "-c",
        "import sys,torch;print('PY', sys.version.split()[0]);"
        "print('TORCH', torch.__version__);"
        "print('CUDA', torch.version.cuda);"
        "print('CXX11ABI', torch.compiled_with_cxx11_abi())"], check=False)
    for pkg in ("flash_attn", "vllm_flash_attn", "flash_attn_2_cuda"):
        subprocess.run(["python3", "-c",
            f"import {pkg};print('HAVE {pkg}', getattr({pkg},'__version__','(no ver)'))"],
            check=False)
    subprocess.run(["pip", "show", "flash-attn"], check=False)
    print("=== VOLUME USAGE ===")
    subprocess.run("df -h /vol; echo '--- /vol/* ---'; du -sh /vol/* 2>/dev/null; "
                   "echo '--- /vol/rl_assets/* ---'; du -sh /vol/rl_assets/* 2>/dev/null",
                   shell=True, check=False)
    print("=== RL PARQUET ROW COUNTS ===")
    subprocess.run(["python3", "-c",
        "import pyarrow.parquet as pq\n"
        "for n,p in [('train','/vol/rl_assets/rl_parquet/train.parquet'),"
        "('test','/vol/rl_assets/rl_parquet/test.parquet')]:\n"
        "    f=pq.ParquetFile(p); print(n,'rows=',f.metadata.num_rows,'cols=',f.schema_arrow.names)"],
        check=False)
    return "diag done"


@app.function(image=image, volumes={"/vol": vol}, timeout=1800)
def prep_resume():
    """Salvage the v1 run: its `best` ckpt (step15, 65.6%) is COMPLETE but the
    regular global_step_10 lost FSDP shards (incomplete commit on hard kill).
    Copy best → global_step_15 + set latest_checkpointed_iteration=15 so verl
    resume_mode=auto loads the clean step-15 state and continues to epoch 40."""
    import subprocess, os
    base = "/vol/rl_assets/rl_k8_v1_out"
    src, dst = f"{base}/best/actor", f"{base}/global_step_15/actor"
    need = f"{dst}/model_world_size_4_rank_0.pt"
    if not os.path.exists(need):
        os.makedirs(f"{base}/global_step_15", exist_ok=True)
        subprocess.run(f"cp -r '{src}' '{dst}'", shell=True, check=True)
    with open(f"{base}/latest_checkpointed_iteration.txt", "w") as f:
        f.write("15")
    vol.commit()
    print("global_step_15/actor:", sorted(os.listdir(dst)))
    return "prepped global_step_15 from best (iter=15)"


@app.function(image=image, gpu="H200:4", cpu=32, volumes={"/vol": vol},
              timeout=24 * 3600,
              # SELF-HEAL: 150 epochs (~26h) > Modal's 24h timeout cap, so the run
              # MUST cross at least one boundary. Modal retries get a FRESH 24h on
              # each attempt (per docs) and also fire on preemption/crash → each retry
              # re-runs train_rl which resume_mode=auto resumes from the latest durable
              # checkpoint. Fully server-side: survives the user's machine being off.
              retries=modal.Retries(max_retries=10, backoff_coefficient=1.0,
                                    initial_delay=30.0),
              # OPENAI_API_KEY (X2 skill_updater; harmless when X2 off) +
              # WANDB_API_KEY (live online training curves, entity medagent).
              secrets=[modal.Secret.from_name("openai-secret"),
                       modal.Secret.from_name("wandb-secret")])
def train_rl(epochs: int = 150, train_size: int = 16, val_size: int = 64,
             group_size: int = 8, test_freq: int = 5):
    import os
    os.environ.update(RL_ENV)
    # editable-install verl fork so `import verl` resolves to our patched copy
    subprocess.run(["pip", "install", "-e", ".", "--no-deps"],
                   cwd="/root/app/SkillRL", check=False)

    actor = "/vol/rl_assets/actor_k8_expanded_vocab"
    # v2: clean restart from the baked actor. v1's checkpoints are CORRUPT — the
    # original run was hard-killed before any vol.commit(), so its big shard files
    # have valid sizes but truncated zip tails (torch.load → "failed finding central
    # directory"). The periodic-commit fix above makes v2's checkpoints durable.
    out = "/vol/rl_assets/rl_k8_v2_out"
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
        # mini-batch = one full update over the rollout batch (train×group
        # trajectories). Hardcoding 128 breaks the smoke (train_size 8 → only 64
        # trajectories < 128 → verl assert). Scale it: full run 16×8=128 unchanged.
        f"actor_rollout_ref.actor.ppo_mini_batch_size={train_size * group_size}",
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
    # DURABILITY: Modal volume writes aren't persisted until vol.commit(); a hard
    # mid-run kill else leaves the in-flight checkpoint with missing FSDP rank shards
    # (exactly what corrupted global_step_10 → resume FileNotFoundError). Commit every
    # 3 min in a background thread so every saved checkpoint becomes durable on its own.
    import threading
    stop_commit = threading.Event()
    def _periodic_commit():
        while not stop_commit.wait(180):
            try:
                vol.commit()
            except Exception as e:
                print("periodic vol.commit warning:", e)
    committer = threading.Thread(target=_periodic_commit, daemon=True)
    committer.start()
    try:
        subprocess.run(cmd, cwd="/root/app/SkillRL", check=True)
    finally:
        stop_commit.set()
        vol.commit()
    return f"RL done (epochs={epochs}) -> {out}"


@app.local_entrypoint()
def main(epochs: int = 150, train_size: int = 16, val_size: int = 64,
         group_size: int = 8, test_freq: int = 5, check: bool = False,
         diag_only: bool = False, prep: bool = False):
    if prep:
        print(prep_resume.remote())
    elif diag_only:
        print(diag.remote())
    elif check:
        print(import_check.remote())
    else:
        print(train_rl.remote(epochs, train_size, val_size, group_size, test_freq))
