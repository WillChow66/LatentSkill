#!/bin/bash
#SBATCH --job-name=composer_sft_k8
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=logs/composer_sft_k8_%j.out
#SBATCH --error=logs/composer_sft_k8_%j.err

# Stage 1 (k=8 ablation): same as composer_sft_3h.sbatch but latents_per_skill=8.
# Tests whether k=2 latent capacity is the bottleneck for TRAINED ≈ UNTRAINED
# at Stage-2-only eval (Stage 1 k=2 produced TRAINED 29/140 ≈ UNTRAINED 33/140).
# k=8 → 4× the bits per skill → check if Composer SFT extracts richer latents
# when given more capacity.
#
# Output dir: composer_sft_k8/  (NEW, does NOT overwrite k=2 composer_sft/)
# Disk: ~7GB total (composer_epoch4 6.5GB + 4× query_latents ~70MB each).
#
# Memory: query_latents 16K params (2× of k=2's 8K), latent_proj unchanged
# (D-only, k-independent), activations slightly larger (n_skills * k positions in
# joint_forward grows from 88 → 352). Still ~88GB / 120GB. Fits.

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

COMPOSER="Qwen/Qwen2.5-3B-Instruct"
ACTOR="Qwen/Qwen2.5-7B-Instruct"
SFT_DATA="/work/nvme/bfdz/xzhou10/huggingface/hub/datasets--Jianwen--SkillRL-SFT-Data/snapshots/bd3996c4e863ac59e6e4ab35549cf3741faf5e4f/alfworld/train-00000-of-00001.parquet"
OUTPUT_DIR="/work/nvme/bfdz/xzhou10/checkpoints/composer_sft_k8"

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
    --latents-per-skill 8 \
    --save-every 10000 \
    --log-every 20 \
    --wandb-run-name "composer_sft_trained_k8_$(date +%Y%m%d)"

echo "Stage 1 Composer SFT (k=8) complete."
ls -la "$OUTPUT_DIR"
