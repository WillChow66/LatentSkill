#!/bin/bash
#SBATCH --job-name=eval_sft_lat
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval_sft_lat_%j.out
#SBATCH --error=logs/eval_sft_lat_%j.err

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

mkdir -p results

SFT_CKPT="/work/nvme/bdns/xzhou10/checkpoints/latent_skills_fullparam/global_step_116"
LATENT_LIB="data/debug_composer/latent_skill_library.pt"

python3 -m src.eval_latent_alfworld \
    --model "$SFT_CKPT" \
    --latent-library "$LATENT_LIB" \
    --num-episodes 140 \
    --max-steps 50 \
    --output results/eval_sft_latent.json

echo "Eval SFT latent complete."
