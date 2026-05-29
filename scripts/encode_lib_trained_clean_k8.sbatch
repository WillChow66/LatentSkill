#!/bin/bash
#SBATCH --job-name=enc_trained_clean_k8
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=logs/enc_trained_clean_k8_%j.out
#SBATCH --error=logs/enc_trained_clean_k8_%j.err

# Stage 2 (TRAINED k=8, CLEAN): encode 44 skills with bug-fixed k=8 Composer.

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

COMPOSER_CKPT="/work/nvme/bfdz/xzhou10/checkpoints/composer_sft_clean_k8/composer_epoch4"
QL_PATH=$(ls -t /work/nvme/bfdz/xzhou10/checkpoints/composer_sft_clean_k8/query_latents_step*.pt | head -1)
ACTOR="Qwen/Qwen2.5-7B-Instruct"
LIBDIR="/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_trained_clean_k8"
EXPANDED_DIR="${LIBDIR}/actor_expanded_vocab"

mkdir -p "$LIBDIR"

echo "=== Encode 44 skills with CLEAN TRAINED Composer (k=8) ==="
python -m src.encode_latent_library \
    --composer-ckpt "$COMPOSER_CKPT" \
    --query-latents "$QL_PATH" \
    --actor-path "$ACTOR" \
    --output-dir "$LIBDIR" \
    --variant TRAINED \
    --latents-per-skill 8 \
    --device cuda

echo "=== Bake into Actor vocab ==="
python -m src.expand_vocab_with_skills expand \
    --sft-checkpoint "$ACTOR" \
    --latent-library "${LIBDIR}/latent_skill_library.pt" \
    --output-dir "$EXPANDED_DIR"

echo "Stage 2 TRAINED clean k=8 complete."
ls -la "$EXPANDED_DIR"
