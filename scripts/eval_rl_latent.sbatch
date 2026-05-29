#!/bin/bash
#SBATCH --job-name=eval_rl_lat
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval_rl_lat_%j.out
#SBATCH --error=logs/eval_rl_lat_%j.err

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

mkdir -p results

# Use v2 ckpt as base (matches what RL was trained on)
V2_CKPT="/work/nvme/bdns/xzhou10/checkpoints/latent_skills_token_v2"
RL_LATENT_LIB="data/latent_library/rl_best_latent_library.pt"

python3 -m src.eval_latent_alfworld \
    --model "$V2_CKPT" \
    --latent-library "$RL_LATENT_LIB" \
    --num-episodes 140 \
    --max-steps 50 \
    --output results/eval_rl_latent.json

echo "Eval Plan C RL latent complete."
