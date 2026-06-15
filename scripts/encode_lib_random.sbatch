#!/bin/bash
#SBATCH --job-name=enc_random
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=logs/enc_random_%j.out
#SBATCH --error=logs/enc_random_%j.err

# Garbage-latent control (4th cell): norm-matched Gaussian noise library,
# baked into vanilla Qwen2.5-7B. Depends on latent_lib_untrained being rebuilt
# first (reference for norms). If RANDOM eval ≈ no-skill (~2%) while UNTRAINED
# = 23.6%, the UNTRAINED gap = real info through the pretrained 3B encoder.

set -e
source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

REF="/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_untrained/latent_skill_library.pt"
LIBDIR="/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_random"
ACTOR="Qwen/Qwen2.5-7B-Instruct"

mkdir -p "$LIBDIR"

echo "=== Step 1: norm-matched random library from UNTRAINED reference ==="
python -m src.make_random_latent_library \
    --reference "$REF" \
    --output "${LIBDIR}/latent_skill_library.pt" \
    --seed 123

echo "=== Step 2: bake into vanilla Actor vocab ==="
python -m src.expand_vocab_with_skills expand \
    --sft-checkpoint "$ACTOR" \
    --latent-library "${LIBDIR}/latent_skill_library.pt" \
    --output-dir "${LIBDIR}/actor_expanded_vocab"

echo "Garbage-latent control bake complete."
ls -la "${LIBDIR}/actor_expanded_vocab"
