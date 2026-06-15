#!/bin/bash
#SBATCH --job-name=joint_sft_B
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/joint_sft_B_%j.out
#SBATCH --error=logs/joint_sft_B_%j.err

# Plan A ablation B: Frozen-random query_latents + train model only.
# Same data/hyperparams/compute as joint_sft_fsdp.sbatch — only diff is
# --freeze-latents flag. Compare downstream eval to measure Composer training value.

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
SFT_DATA="/work/nvme/bfdz/xzhou10/huggingface/hub/datasets--Jianwen--SkillRL-SFT-Data/snapshots/bd3996c4e863ac59e6e4ab35549cf3741faf5e4f/alfworld/train-00000-of-00001.parquet"
OUTPUT_DIR="/work/nvme/bfdz/xzhou10/checkpoints/joint_sft_B_frozen"

mkdir -p "$OUTPUT_DIR"

accelerate launch \
    --num_processes 4 \
    --use_fsdp \
    --mixed_precision bf16 \
    --fsdp_sharding_strategy SHARD_GRAD_OP \
    --fsdp_state_dict_type FULL_STATE_DICT \
    --fsdp_offload_params false \
    -m src.train_joint_sft_fsdp \
    --model-path "$MODEL_PATH" \
    --sft-data "$SFT_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 4 \
    --model-lr 1e-5 \
    --weight-decay 0.01 \
    --warmup-ratio 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --max-length 4096 \
    --save-every 10000 \
    --log-every 20 \
    --freeze-latents

echo "Joint SFT B (frozen-random ablation) complete."
