#!/bin/bash
#SBATCH --job-name=eval_rl_final
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_rl_final_%j.out
#SBATCH --error=logs/eval_rl_final_%j.err

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

mkdir -p results data/latent_library

V2_CKPT="/work/nvme/bdns/xzhou10/checkpoints/latent_skills_token_v2"
FINAL_QL="/work/nvme/bfdz/xzhou10/checkpoints/rl_e2e_full/global_step_150/query_latents_step150.pt"
LATENT_LIB="data/latent_library/rl_final_step150_library.pt"

# Step 1: Encode skill library using final RL ckpt query_latents
python3 -m src.encode_rl_latent_library \
    --model "$V2_CKPT" \
    --query-latents "$FINAL_QL" \
    --output "$LATENT_LIB" \
    --latents-per-skill 2

# Step 2: Eval on ALFWorld
python3 -m src.eval_latent_alfworld \
    --model "$V2_CKPT" \
    --latent-library "$LATENT_LIB" \
    --num-episodes 140 \
    --max-steps 50 \
    --output results/eval_rl_latent_final.json

echo "Eval Plan C RL (final step 150) complete."
