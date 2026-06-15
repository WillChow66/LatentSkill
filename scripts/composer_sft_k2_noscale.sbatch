#!/bin/bash
#SBATCH --job-name=composer_noscale
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=logs/composer_noscale_%j.out
#SBATCH --error=logs/composer_noscale_%j.err

# Init-scale ablation: identical to clean k=2 EXCEPT query_latents init uses
# UNSCALED randn (LatentMem-style, --ql-init-scale 1.0, norm ~45) instead of
# our *0.02 (norm ~0.9, matches Qwen2.5-3B embedding std). Tests whether the
# 0.02 scaling matters. Output: composer_sft_clean_k2_noscale/.

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

COMPOSER="Qwen/Qwen2.5-3B-Instruct"
ACTOR="Qwen/Qwen2.5-7B-Instruct"
SFT_DATA="/work/nvme/bfdz/xzhou10/huggingface/hub/datasets--Jianwen--SkillRL-SFT-Data/snapshots/bd3996c4e863ac59e6e4ab35549cf3741faf5e4f/alfworld/train-00000-of-00001.parquet"
OUTPUT_DIR="/work/nvme/bfdz/xzhou10/checkpoints/composer_sft_clean_k2_noscale"

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
    --ql-init-scale 1.0 \
    --save-every 10000 \
    --log-every 20 \
    --wandb-run-name "composer_sft_k2_noscale_$(date +%Y%m%d)"

echo "Init-scale ablation SFT (k=2, unscaled) complete."
ls -la "$OUTPUT_DIR"
