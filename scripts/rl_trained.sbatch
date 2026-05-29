#!/bin/bash
#SBATCH --job-name=rl_trained
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=logs/rl_trained_%j.out
#SBATCH --error=logs/rl_trained_%j.err

# Stage 3 (TRAINED variant): RL on Actor with TRAINED-Composer-encoded latents
# in vocab, + dynamic skill bank using TRAINED Composer for live encoding.
#
# DEPENDENCIES (must complete first):
#   - Stage 1: composer_sft.sbatch (Job 2197534)  → composer_sft/composer_epoch4
#   - Stage 2: encode_lib_trained.sbatch          → latent_lib_trained/actor_expanded_vocab
#
# Differences vs old Plan C run_rl_full_param_x2.sbatch:
#   - MODEL_PATH points to the new vocab-expanded Actor (44 base SKILL tokens)
#   - COMPOSER_PATH = TRAINED 3B Composer ckpt (was 7B in Plan C — verl/skill_composer.py
#     may need verification for 3B + projection layer)
#   - COMPOSER_QL contains query_latents AND latent_proj_state in one file

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill/SkillRL
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_memory_monitor_refresh_ms=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY not set (required for X2 dynamic skill bank)"
    exit 1
fi

MODEL_PATH="/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_trained/actor_expanded_vocab"
COMPOSER_PATH="/work/nvme/bfdz/xzhou10/checkpoints/composer_sft/composer_epoch4"
COMPOSER_QL=$(ls -t /work/nvme/bfdz/xzhou10/checkpoints/composer_sft/query_latents_step*.pt | head -1)
OUTPUT_DIR="/work/nvme/bfdz/xzhou10/checkpoints/rl_trained_composer"

# Sanity checks
for p in "$MODEL_PATH" "$COMPOSER_PATH"; do
    if [ ! -d "$p" ]; then
        echo "ERROR: missing $p — run prerequisite stages first"
        exit 1
    fi
done
if [ ! -f "$COMPOSER_QL" ]; then
    echo "ERROR: missing $COMPOSER_QL"
    exit 1
fi

num_cpus_per_env_worker=0.1
train_data_size=16
val_data_size=64
group_size=8

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    +env.use_skills_only_memory=True \
    +env.skills_only_memory.skills_json_path=memory_data/alfworld/claude_style_skills.json \
    +env.skills_only_memory.top_k=6 \
    +env.skills_only_memory.latent_token_mode=True \
    +env.skills_only_memory.enable_dynamic_update=True \
    +env.skills_only_memory.update_threshold=0.4 \
    +env.skills_only_memory.max_new_skills=3 \
    +env.skills_only_memory.composer_model_path=$COMPOSER_PATH \
    +env.skills_only_memory.composer_query_latents_path=$COMPOSER_QL \
    +env.skills_only_memory.composer_variant=TRAINED \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='latentskill' \
    trainer.experiment_name='rl_trained_composer_'$(date +%Y%m%d) \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    trainer.max_actor_ckpt_to_keep=2
