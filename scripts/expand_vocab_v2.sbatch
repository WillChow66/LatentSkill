#!/bin/bash
#SBATCH --job-name=expand_v2
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/expand_v2_%j.out
#SBATCH --error=logs/expand_v2_%j.err

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

# Regenerate checkpoint with:
# 1. Trained Composer latent library (from Stage 1)
# 2. Untied LM head (SKILL output weights = 0)
python3 -m src.expand_vocab_with_skills expand \
    --sft-checkpoint /work/nvme/bdns/xzhou10/checkpoints/latent_skills_fullparam/global_step_116 \
    --latent-library data/debug_composer/latent_skill_library.pt \
    --output-dir /work/nvme/bdns/xzhou10/checkpoints/latent_skills_token_v2
