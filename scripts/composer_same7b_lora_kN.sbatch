#!/bin/bash
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=04:00:00
# B-arm: LoRA composer (LatentMem-style) | same protocol as full-FT main batch.
# composer = actor = Qwen2.5-7B; only LoRA adapters + ql train; actor frozen.
# Saved ckpt is LoRA-merged standard HF (encode-compatible).
set -e
: "${K:?K env var must be set}"
source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1
MODEL7B="Qwen/Qwen2.5-7B-Instruct"
SFT_DATA="/work/nvme/bfdz/xzhou10/huggingface/hub/datasets--Jianwen--SkillRL-SFT-Data/snapshots/bd3996c4e863ac59e6e4ab35549cf3741faf5e4f/alfworld/train-00000-of-00001.parquet"
OUTPUT_DIR="/work/nvme/bfdz/xzhou10/checkpoints/composer_same7b_lora_k${K}"
mkdir -p "$OUTPUT_DIR"
accelerate launch --num_processes 4 --mixed_precision bf16 \
    -m src.train_composer_only \
    --composer-path "$MODEL7B" --actor-path "$MODEL7B" \
    --sft-data "$SFT_DATA" --output-dir "$OUTPUT_DIR" \
    --epochs 4 --composer-lr 1e-4 --latent-lr 5e-3 \
    --weight-decay 0.01 --warmup-ratio 0.1 --adam-beta1 0.9 --adam-beta2 0.95 \
    --max-length 4096 --latents-per-skill ${K} --seed 42 \
    --use-lora --lora-r 16 --lora-alpha 32 --lora-dropout 0.1 --lora-targets q_proj,v_proj \
    --save-every 10000 --log-every 50 \
    --wandb-run-name "lora7b_k${K}_$(date +%Y%m%d)"
echo "LoRA train k=${K} done."; ls -la "$OUTPUT_DIR"
