#!/bin/bash
#SBATCH --job-name=joint_path2_un
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=logs/joint_path2_un_%j.out
#SBATCH --error=logs/joint_path2_un_%j.err

# Path 2 UNTRAINED ablation: Composer body + ql + proj FROZEN at random init,
# only Actor trains. Apples-to-apples control vs joint_path2_trained to test
# "does training the Composer system add value beyond Actor adaptation?"
# Output: joint_path2_untrained/  (only actor_epoch4/ + ql files; no composer ckpt)

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

COMPOSER="Qwen/Qwen2.5-3B-Instruct"
ACTOR="Qwen/Qwen2.5-7B-Instruct"
SFT_DATA="/work/nvme/bdns/xzhou10/huggingface/hub/datasets--Jianwen--SkillRL-SFT-Data/snapshots/bd3996c4e863ac59e6e4ab35549cf3741faf5e4f/alfworld/train-00000-of-00001.parquet"
OUTPUT_DIR="/work/nvme/bfdz/xzhou10/checkpoints/joint_path2_untrained"
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
    --epochs 4 \
    --actor-lr 1e-5 \
    --max-length 4096 \
    --latents-per-skill 2 \
    --freeze-composer \
    --save-every 10000 \
    --log-every 20 \
    --wandb-run-name "joint_path2_untrained_$(date +%Y%m%d)"

echo "Path 2 UNTRAINED full SFT complete."
ls -la "$OUTPUT_DIR"
