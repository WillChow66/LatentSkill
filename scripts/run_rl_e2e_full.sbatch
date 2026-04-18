#!/bin/bash
#SBATCH --job-name=rl_e2e_full
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=logs/rl_e2e_full_%j.out
#SBATCH --error=logs/rl_e2e_full_%j.err

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill/SkillRL
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_memory_monitor_refresh_ms=0

# V2 checkpoint: untied LM head, SKILL output weights = 0
export MODEL_PATH="/work/nvme/bdns/xzhou10/checkpoints/latent_skills_token_v2"
SKILL_TOKEN_MAP="$MODEL_PATH/skill_token_map.json"
SKILLS_JSON="memory_data/alfworld/claude_style_skills.json"
PRETRAINED_QL="/u/xzhou10/latentskill/data/debug_composer/trained_query_latents.pt"

num_cpus_per_env_worker=0.1
train_data_size=16
val_data_size=64
group_size=8

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

# Full RL: 150 epochs, matches SkillRL paper hyperparameters
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
    +actor_rollout_ref.composer.latents_per_skill=2 \
    +actor_rollout_ref.composer.pretrained_query_latents=$PRETRAINED_QL \
    +actor_rollout_ref.composer.skills_json_path=$SKILLS_JSON \
    +actor_rollout_ref.composer.skill_token_map_path=$SKILL_TOKEN_MAP \
    +actor_rollout_ref.composer.lr=5e-3 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    +env.use_skills_only_memory=True \
    +env.skills_only_memory.skills_json_path=$SKILLS_JSON \
    +env.skills_only_memory.top_k=6 \
    +env.skills_only_memory.latent_token_mode=True \
    +env.skills_only_memory.enable_dynamic_update=False \
    trainer.critic_warmup=0 \
    "trainer.logger=[console,wandb]" \
    trainer.project_name=latentskill \
    trainer.experiment_name=rl_e2e_composer_full \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    trainer.default_local_dir=/work/nvme/bfdz/xzhou10/checkpoints/rl_e2e_full \
    trainer.default_hdfs_dir=null
