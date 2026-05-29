#!/bin/bash
#SBATCH --job-name=encode_rl
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/encode_rl_%j.out
#SBATCH --error=logs/encode_rl_%j.err

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

V2_CKPT="/work/nvme/bdns/xzhou10/checkpoints/latent_skills_token_v2"
BEST_QL="/work/nvme/bfdz/xzhou10/checkpoints/rl_e2e_full/best/query_latents_step40.pt"
OUTPUT="data/latent_library/rl_best_latent_library.pt"

python3 -m src.encode_rl_latent_library \
    --model "$V2_CKPT" \
    --query-latents "$BEST_QL" \
    --output "$OUTPUT" \
    --latents-per-skill 2

ls -la "$OUTPUT"
echo "Encode complete."
