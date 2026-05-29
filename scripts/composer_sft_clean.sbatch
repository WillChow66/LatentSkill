#!/bin/bash
#SBATCH --job-name=composer_clean
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=logs/composer_clean_%j.out
#SBATCH --error=logs/composer_clean_%j.err

# Stage 1 SFT (k=2) — CLEAN version with embed_tokens-frozen bug FIXED.
# Trainer now untie tie_word_embeddings BEFORE freezing lm_head, so
# embed_tokens stays trainable. Output: composer_sft_clean/ (NEW dir, does NOT
# overwrite composer_sft/ which has the bug — kept for comparison).

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

COMPOSER="Qwen/Qwen2.5-3B-Instruct"
ACTOR="Qwen/Qwen2.5-7B-Instruct"
SFT_DATA="/work/nvme/bdns/xzhou10/huggingface/hub/datasets--Jianwen--SkillRL-SFT-Data/snapshots/bd3996c4e863ac59e6e4ab35549cf3741faf5e4f/alfworld/train-00000-of-00001.parquet"
OUTPUT_DIR="/work/nvme/bfdz/xzhou10/checkpoints/composer_sft_clean"

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
    --wandb-run-name "composer_sft_clean_k2_$(date +%Y%m%d)"

echo "Stage 1 Composer SFT (clean k=2) complete."
ls -la "$OUTPUT_DIR"
