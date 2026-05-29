#!/bin/bash
#SBATCH --job-name=merge_rl
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --output=logs/merge_rl_%j.out
#SBATCH --error=logs/merge_rl_%j.err

source /u/xzhou10/latentskill/scripts/setup_env.sh
export PYTHONUNBUFFERED=1

ACTOR_DIR="/work/nvme/bfdz/xzhou10/huggingface/hub/models--Jianwen--Alfworld-7B-RL/snapshots/2ce16cb90e6357892dde201928279d4513d35c59/actor"
TARGET_DIR="/work/nvme/bfdz/xzhou10/checkpoints/skillrl_alfworld_rl_hf"

mkdir -p "$TARGET_DIR"

cd /u/xzhou10/latentskill
python3 -m src.merge_fsdp_to_hf \
    --actor-dir "$ACTOR_DIR" \
    --target-dir "$TARGET_DIR"

ls -la "$TARGET_DIR"
echo "Merge complete."
