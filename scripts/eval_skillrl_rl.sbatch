#!/bin/bash
#SBATCH --job-name=eval_skillrl_rl
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval_skillrl_rl_%j.out
#SBATCH --error=logs/eval_skillrl_rl_%j.err

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

mkdir -p results

# Merged HF format from Jianwen/Alfworld-7B-RL FSDP shards
SKILLRL_RL_CKPT="/work/nvme/bfdz/xzhou10/checkpoints/skillrl_alfworld_rl_hf"

python3 -m src.eval_text_alfworld \
    --model "$SKILLRL_RL_CKPT" \
    --num-episodes 140 \
    --max-steps 50 \
    --top-k 6 \
    --output results/eval_skillrl_rl.json

echo "Eval SkillRL RL complete."
