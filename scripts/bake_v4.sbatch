#!/bin/bash
#SBATCH --job-name=bake_v4
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/bake_v4_%j.out
#SBATCH --error=logs/bake_v4_%j.err

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

V3_CKPT="/work/nvme/bfdz/xzhou10/checkpoints/latent_skills_token_v3"
V4_OUT="/work/nvme/bfdz/xzhou10/checkpoints/latent_skills_token_v4"

python3 -m src.bake_sft_latents_into_v4 \
    --v3-ckpt "$V3_CKPT" \
    --output-dir "$V4_OUT" \
    --num-dyn-slots 100 \
    --latents-per-skill 2

ls -la "$V4_OUT" | head -20
echo "Bake v4 complete."
