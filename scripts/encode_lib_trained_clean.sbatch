#!/bin/bash
#SBATCH --job-name=enc_trained_clean
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=logs/enc_trained_clean_%j.out
#SBATCH --error=logs/enc_trained_clean_%j.err

# Stage 2 (TRAINED k=2, CLEAN): encode 44 skills with bug-fixed Composer.
# Output: latent_lib_trained_clean/.

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

COMPOSER_CKPT="/work/nvme/bfdz/xzhou10/checkpoints/composer_sft_clean/composer_epoch4"
QL_PATH=$(ls -t /work/nvme/bfdz/xzhou10/checkpoints/composer_sft_clean/query_latents_step*.pt | head -1)
ACTOR="Qwen/Qwen2.5-7B-Instruct"
LIBDIR="/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_trained_clean"
EXPANDED_DIR="${LIBDIR}/actor_expanded_vocab"

mkdir -p "$LIBDIR"

echo "=== Encode 44 skills with CLEAN TRAINED Composer (k=2) ==="
python -m src.encode_latent_library \
    --composer-ckpt "$COMPOSER_CKPT" \
    --query-latents "$QL_PATH" \
    --actor-path "$ACTOR" \
    --output-dir "$LIBDIR" \
    --variant TRAINED \
    --latents-per-skill 2 \
    --device cuda

echo "=== Bake into Actor vocab ==="
python -m src.expand_vocab_with_skills expand \
    --sft-checkpoint "$ACTOR" \
    --latent-library "${LIBDIR}/latent_skill_library.pt" \
    --output-dir "$EXPANDED_DIR"

echo "Stage 2 TRAINED clean k=2 complete."
ls -la "$EXPANDED_DIR"
