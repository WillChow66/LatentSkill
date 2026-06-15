#!/bin/bash
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=04:00:00

# K-sweep template. Submit with:
#   sbatch --export=ALL,K=1 --job-name=composer_k1 \
#          -o logs/composer_k1_%j.out -e logs/composer_k1_%j.err \
#          scripts/composer_sft_clean_kN.sbatch
# Reads K from env (must be set). Output: composer_sft_clean_k${K}/.

set -e
: "${K:?K env var must be set, e.g. --export=ALL,K=3}"

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

COMPOSER="Qwen/Qwen2.5-3B-Instruct"
ACTOR="Qwen/Qwen2.5-7B-Instruct"
SFT_DATA="/work/nvme/bfdz/xzhou10/huggingface/hub/datasets--Jianwen--SkillRL-SFT-Data/snapshots/bd3996c4e863ac59e6e4ab35549cf3741faf5e4f/alfworld/train-00000-of-00001.parquet"
OUTPUT_DIR="/work/nvme/bfdz/xzhou10/checkpoints/composer_sft_clean_k${K}"

mkdir -p "$OUTPUT_DIR"
echo "=== Stage 1 SFT k=${K} (Path 1 Full FT) ==="
echo "Output: $OUTPUT_DIR"

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
    --latents-per-skill ${K} \
    --save-every 10000 \
    --log-every 20 \
    --wandb-run-name "composer_sft_clean_k${K}_$(date +%Y%m%d)"

echo "Stage 1 SFT k=${K} done."
ls -la "$OUTPUT_DIR"
