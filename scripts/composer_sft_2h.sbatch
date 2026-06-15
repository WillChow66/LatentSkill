#!/bin/bash
#SBATCH --job-name=composer_sft
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/composer_sft_3h_%j.out
#SBATCH --error=logs/composer_sft_3h_%j.err

# Stage 1 (3h budget — same code/HP as 8h composer_sft.sbatch).
# Mini run (500 sample × 4 epoch) took 6:53 min @ 0.83 sec/step.
# Full run = 7500 sample × 4 epoch / 4 rank = 7500 steps × 0.83 = ~104 min.
# 3h budget = ~50% margin. Backfill ~2.7× faster than 8h budget version.
# Same OUTPUT_DIR as 8h variant — only ONE will actually run; the loser is
# cancelled by hand when winner starts (or its dependency target downstream
# auto-triggers off whichever finishes first).

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

COMPOSER="Qwen/Qwen2.5-3B-Instruct"
ACTOR="Qwen/Qwen2.5-7B-Instruct"
SFT_DATA="/work/nvme/bfdz/xzhou10/huggingface/hub/datasets--Jianwen--SkillRL-SFT-Data/snapshots/bd3996c4e863ac59e6e4ab35549cf3741faf5e4f/alfworld/train-00000-of-00001.parquet"
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

echo "Stage 1 Composer SFT (3h variant) complete."
ls -la "$OUTPUT_DIR"
