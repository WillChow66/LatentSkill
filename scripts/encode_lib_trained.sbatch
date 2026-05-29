#!/bin/bash
#SBATCH --job-name=enc_trained
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=logs/enc_trained_%j.out
#SBATCH --error=logs/enc_trained_%j.err

# Stage 2 (TRAINED variant): encode 44 ALFWorld skills with Stage-1-trained
# Composer + query_latents + latent_proj. Then bake into Actor vocab.

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

# Inputs from Stage 1 (Job 2197534's output)
COMPOSER_CKPT="/work/nvme/bfdz/xzhou10/checkpoints/composer_sft/composer_epoch4"
QL_PATH=$(ls -t /work/nvme/bfdz/xzhou10/checkpoints/composer_sft/query_latents_step*.pt | head -1)
ACTOR="Qwen/Qwen2.5-7B-Instruct"

LIBDIR="/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_trained"
EXPANDED_DIR="${LIBDIR}/actor_expanded_vocab"

mkdir -p "$LIBDIR"

echo "=== Step 1: encode 44 skills with TRAINED Composer ==="
echo "Composer: $COMPOSER_CKPT"
echo "QL:       $QL_PATH"
python -m src.encode_latent_library \
    --composer-ckpt "$COMPOSER_CKPT" \
    --query-latents "$QL_PATH" \
    --actor-path "$ACTOR" \
    --output-dir "$LIBDIR" \
    --variant TRAINED \
    --latents-per-skill 2 \
    --device cuda

echo
echo "=== Step 2: bake latent vectors into Actor vocab embedding ==="
python -m src.expand_vocab_with_skills expand \
    --sft-checkpoint "$ACTOR" \
    --latent-library "${LIBDIR}/latent_skill_library.pt" \
    --output-dir "$EXPANDED_DIR"

echo
echo "Stage 2 TRAINED variant complete."
ls -la "$EXPANDED_DIR"
