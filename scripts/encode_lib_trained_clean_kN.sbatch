#!/bin/bash
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:30:00

# K-sweep encode template. Reads K from env. Bakes into vanilla Qwen2.5-7B
# (Path 1 Full FT — actor was frozen during SFT, so RL starts from vanilla 7B).
# Output: latent_lib_trained_clean_k${K}/.

set -e
: "${K:?K env var must be set}"

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

COMPOSER_CKPT="/work/nvme/bfdz/xzhou10/checkpoints/composer_sft_clean_k${K}/composer_epoch4"
QL_PATH=$(ls -t /work/nvme/bfdz/xzhou10/checkpoints/composer_sft_clean_k${K}/query_latents_step*.pt | head -1)
ACTOR="Qwen/Qwen2.5-7B-Instruct"
LIBDIR="/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_trained_clean_k${K}"
EXPANDED_DIR="${LIBDIR}/actor_expanded_vocab"

mkdir -p "$LIBDIR"

echo "=== Encode 44 skills (k=${K}) ==="
python -m src.encode_latent_library \
    --composer-ckpt "$COMPOSER_CKPT" \
    --query-latents "$QL_PATH" \
    --actor-path "$ACTOR" \
    --output-dir "$LIBDIR" \
    --variant TRAINED \
    --latents-per-skill ${K} \
    --device cuda

echo "=== Bake into vanilla Qwen2.5-7B vocab ==="
python -m src.expand_vocab_with_skills expand \
    --sft-checkpoint "$ACTOR" \
    --latent-library "${LIBDIR}/latent_skill_library.pt" \
    --output-dir "$EXPANDED_DIR"

echo "Stage 2 k=${K} done."
ls -la "$EXPANDED_DIR"
