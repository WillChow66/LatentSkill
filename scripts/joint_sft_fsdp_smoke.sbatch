#!/bin/bash
#SBATCH --job-name=joint_smoke
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --output=logs/joint_smoke_%j.out
#SBATCH --error=logs/joint_smoke_%j.err

# Smoke test: FSDP with attach_query_latents architecture (no outer wrapper).
# Verifies (1) forward+backward, (2) decoder weights actually train and SAVE
# (logged ref_pre_train vs post-train q_proj norm — must differ).
# 100 samples, 1 epoch, ~3-8 minutes.

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
SFT_DATA="/work/nvme/bfdz/xzhou10/huggingface/hub/datasets--Jianwen--SkillRL-SFT-Data/snapshots/bd3996c4e863ac59e6e4ab35549cf3741faf5e4f/alfworld/train-00000-of-00001.parquet"
OUTPUT_DIR="/work/nvme/bfdz/xzhou10/checkpoints/joint_smoke"

mkdir -p "$OUTPUT_DIR"

accelerate launch \
    --num_processes 4 \
    --use_fsdp \
    --mixed_precision bf16 \
    --fsdp_sharding_strategy FULL_SHARD \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap Qwen2DecoderLayer \
    --fsdp_state_dict_type FULL_STATE_DICT \
    --fsdp_use_orig_params true \
    --fsdp_offload_params false \
    -m src.train_joint_sft_fsdp \
    --model-path "$MODEL_PATH" \
    --sft-data "$SFT_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 1 \
    --max-samples 40 \
    --model-lr 1e-5 \
    --latent-lr 5e-3 \
    --max-length 4096 \
    --save-every 10000 \
    --log-every 5

echo "Smoke test complete."
ls -la "$OUTPUT_DIR"
