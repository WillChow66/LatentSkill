#!/bin/bash
#SBATCH --job-name=joint_sft_A3
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/joint_sft_A3_%j.out
#SBATCH --error=logs/joint_sft_A3_%j.err

# Plan A: End-to-end joint SFT (query_latents + 7B model trained together).
# 4 GPUs FSDP via Accelerate. ~36h budget for 3 epochs on 7486 samples.

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
SFT_DATA="/work/nvme/bdns/xzhou10/huggingface/hub/datasets--Jianwen--SkillRL-SFT-Data/snapshots/bd3996c4e863ac59e6e4ab35549cf3741faf5e4f/alfworld/train-00000-of-00001.parquet"
OUTPUT_DIR="/work/nvme/bfdz/xzhou10/checkpoints/joint_sft_A3"

mkdir -p "$OUTPUT_DIR"

accelerate launch \
    --num_processes 4 \
    --use_fsdp \
    --mixed_precision bf16 \
    --fsdp_sharding_strategy FULL_SHARD \
    --fsdp_state_dict_type FULL_STATE_DICT \
    --fsdp_offload_params false \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap Qwen2DecoderLayer \
    -m src.train_joint_sft_fsdp \
    --model-path "$MODEL_PATH" \
    --sft-data "$SFT_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 4 \
    --model-lr 1e-5 \
    --latent-lr 5e-4 \
    --weight-decay 0.01 \
    --warmup-ratio 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --max-length 4096 \
    --save-every 10000 \
    --log-every 20

echo "Joint SFT A complete."
