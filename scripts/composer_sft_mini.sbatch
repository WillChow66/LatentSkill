#!/bin/bash
#SBATCH --job-name=comp_mini
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --output=logs/comp_mini_%j.out
#SBATCH --error=logs/comp_mini_%j.err

# Mini Stage 1 verification: 500 samples × 4 epochs = ~500 steps total.
# Full cosine schedule (warmup 50, decay 450) — long enough to see real
# loss decrease + cumulative q_proj norm movement (>1e-3 expected).
# Cheap pre-check before committing 8h × 4 GPU on full run.

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

COMPOSER="Qwen/Qwen2.5-3B-Instruct"
ACTOR="Qwen/Qwen2.5-7B-Instruct"
SFT_DATA="/work/nvme/bdns/xzhou10/huggingface/hub/datasets--Jianwen--SkillRL-SFT-Data/snapshots/bd3996c4e863ac59e6e4ab35549cf3741faf5e4f/alfworld/train-00000-of-00001.parquet"
OUTPUT_DIR="/work/nvme/bfdz/xzhou10/checkpoints/composer_mini"

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
    --max-samples 500 \
    --composer-lr 1e-5 \
    --latent-lr 5e-3 \
    --weight-decay 0.01 \
    --warmup-ratio 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --max-length 4096 \
    --latents-per-skill 2 \
    --save-every 10000 \
    --log-every 10 \
    --wandb-run-name "composer_mini_$(date +%Y%m%d_%H%M)"

echo "Mini Stage 1 complete."
ls -la "$OUTPUT_DIR"
