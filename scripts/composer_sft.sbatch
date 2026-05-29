#!/bin/bash
#SBATCH --job-name=composer_sft
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=logs/composer_sft_%j.out
#SBATCH --error=logs/composer_sft_%j.err

# Stage 1 (Plan A redesign): Composer SFT.
#   Composer = Qwen2.5-3B-Instruct (DDP, trainable)
#   Actor    = Qwen2.5-7B-Instruct (per-rank replica, FROZEN, eval mode)
#   query_latents = standalone trainable nn.Parameter (k=2, D=2048 for 3B Composer)
# 4 epochs × ~7500 samples / 4 ranks = ~7500 steps total. ~6h budget.

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

COMPOSER="Qwen/Qwen2.5-3B-Instruct"
ACTOR="Qwen/Qwen2.5-7B-Instruct"
SFT_DATA="/work/nvme/bdns/xzhou10/huggingface/hub/datasets--Jianwen--SkillRL-SFT-Data/snapshots/bd3996c4e863ac59e6e4ab35549cf3741faf5e4f/alfworld/train-00000-of-00001.parquet"
OUTPUT_DIR="/work/nvme/bfdz/xzhou10/checkpoints/composer_sft"

mkdir -p "$OUTPUT_DIR"

accelerate launch \
    --num_processes 4 \
    --mixed_precision bf16 \
    -m src.train_composer_only \
    --composer-path "$COMPOSER" \
    --actor-path "$ACTOR" \
    --sft-data "$SFT_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 4 \
    --composer-lr 1e-5 \
    --latent-lr 5e-3 \
    --weight-decay 0.01 \
    --warmup-ratio 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --max-length 4096 \
    --latents-per-skill 2 \
    --save-every 10000 \
    --log-every 20 \
    --wandb-run-name "composer_sft_trained_$(date +%Y%m%d)"

echo "Stage 1 Composer SFT complete."
ls -la "$OUTPUT_DIR"
