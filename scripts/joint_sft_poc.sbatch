#!/bin/bash
#SBATCH --job-name=joint_poc
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=logs/joint_poc_%j.out
#SBATCH --error=logs/joint_poc_%j.err

# Path 2 SFT smoke test (40 samples × 1 epoch, ~5-10 min actual).
# Verifies: (1) DeepSpeed ZeRO-2 wraps both Composer + Actor without crash,
# (2) backward updates BOTH models (composer_qproj + actor_qproj both move),
# (3) save round-trip works for both models.

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

COMPOSER="Qwen/Qwen2.5-3B-Instruct"
ACTOR="Qwen/Qwen2.5-7B-Instruct"
SFT_DATA="/work/nvme/bdns/xzhou10/huggingface/hub/datasets--Jianwen--SkillRL-SFT-Data/snapshots/bd3996c4e863ac59e6e4ab35549cf3741faf5e4f/alfworld/train-00000-of-00001.parquet"
OUTPUT_DIR="/work/nvme/bfdz/xzhou10/checkpoints/joint_path2_smoke"
DS_CONFIG="/u/xzhou10/latentskill/scripts/ds_config_zero2.json"

mkdir -p "$OUTPUT_DIR"

accelerate launch \
    --num_processes 4 \
    --use_deepspeed \
    --deepspeed_config_file "$DS_CONFIG" \
    --zero_stage 2 \
    -m src.train_composer_actor_joint \
    --composer-path "$COMPOSER" \
    --actor-path "$ACTOR" \
    --sft-data "$SFT_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 1 \
    --max-samples 40 \
    --composer-lr 1e-5 \
    --actor-lr 1e-5 \
    --latent-lr 5e-3 \
    --proj-lr 5e-4 \
    --max-length 4096 \
    --latents-per-skill 8 \
    --save-every 10000 \
    --log-every 5 \
    --wandb-run-name "joint_path2_poc_$(date +%Y%m%d_%H%M)"

echo "Path 2 smoke done."
ls -la "$OUTPUT_DIR"
