#!/bin/bash
#SBATCH --job-name=bake_v3
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/bake_v3_%j.out
#SBATCH --error=logs/bake_v3_%j.err

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

V2_CKPT="/work/nvme/bdns/xzhou10/checkpoints/latent_skills_token_v2"
SFT_QL="/u/xzhou10/latentskill/data/debug_composer/trained_query_latents.pt"
V3_OUT="/work/nvme/bfdz/xzhou10/checkpoints/latent_skills_token_v3"

python3 -m src.bake_sft_latents_into_v3 \
    --v2-ckpt "$V2_CKPT" \
    --query-latents "$SFT_QL" \
    --output-dir "$V3_OUT" \
    --latents-per-skill 2

ls -la "$V3_OUT"
echo "Bake complete."
