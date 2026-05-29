#!/bin/bash
#SBATCH --job-name=comp_smoke
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --output=logs/comp_smoke_%j.out
#SBATCH --error=logs/comp_smoke_%j.err

# Stage 1 smoke: 40 samples, 1 epoch. ~3-5 min.
# Verifies: (a) Composer 3B trains via DDP (q_proj norm changes), (b) Actor 7B
# stays bit-identical (frozen check), (c) query_latents trains, (d) save round-trip.

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

COMPOSER="Qwen/Qwen2.5-3B-Instruct"
ACTOR="Qwen/Qwen2.5-7B-Instruct"
SFT_DATA="/work/nvme/bdns/xzhou10/huggingface/hub/datasets--Jianwen--SkillRL-SFT-Data/snapshots/bd3996c4e863ac59e6e4ab35549cf3741faf5e4f/alfworld/train-00000-of-00001.parquet"
OUTPUT_DIR="/work/nvme/bfdz/xzhou10/checkpoints/composer_smoke"

mkdir -p "$OUTPUT_DIR"

# DDP launch: no --use_fsdp (default = multi-GPU DDP for the trainable composer)
accelerate launch \
    --num_processes 4 \
    --mixed_precision bf16 \
    -m src.train_composer_only \
    --composer-path "$COMPOSER" \
    --actor-path "$ACTOR" \
    --sft-data "$SFT_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 1 \
    --max-samples 40 \
    --composer-lr 1e-5 \
    --latent-lr 5e-3 \
    --max-length 4096 \
    --save-every 10000 \
    --log-every 5 \
    --wandb-run-name "composer_smoke_$(date +%Y%m%d_%H%M)"

echo "Smoke complete."
ls -la "$OUTPUT_DIR"
