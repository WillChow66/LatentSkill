#!/bin/bash
#SBATCH --job-name=enc_untrained
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=logs/enc_untrained_%j.out
#SBATCH --error=logs/enc_untrained_%j.err

# Stage 2 (UNTRAINED variant, ablation control): encode 44 ALFWorld skills with
# raw Qwen2.5-3B-Instruct + RANDOM query_latents + RANDOM latent_proj
# (same init seed as Stage 1 → fair comparison). Then bake into Actor vocab.

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

UNTRAINED_COMPOSER="Qwen/Qwen2.5-3B-Instruct"
ACTOR="Qwen/Qwen2.5-7B-Instruct"

LIBDIR="/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_untrained"
EXPANDED_DIR="${LIBDIR}/actor_expanded_vocab"

mkdir -p "$LIBDIR"

echo "=== Step 1: encode 44 skills with UNTRAINED Composer (random ql + proj) ==="
python -m src.encode_latent_library \
    --composer-ckpt "$UNTRAINED_COMPOSER" \
    --actor-path "$ACTOR" \
    --output-dir "$LIBDIR" \
    --variant UNTRAINED \
    --latents-per-skill 2 \
    --seed 42 \
    --device cuda

echo
echo "=== Step 2: bake latent vectors into Actor vocab embedding ==="
python -m src.expand_vocab_with_skills expand \
    --sft-checkpoint "$ACTOR" \
    --latent-library "${LIBDIR}/latent_skill_library.pt" \
    --output-dir "$EXPANDED_DIR"

echo
echo "Stage 2 UNTRAINED variant complete."
ls -la "$EXPANDED_DIR"
