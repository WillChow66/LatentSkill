#!/bin/bash
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:30:00

# Same-model k-sweep Stage 2: encode 44 skills with the trained 7B composer,
# bake into vanilla 7B (actor was frozen). D==D -> no projection in library.

set -e
: "${K:?K env var must be set}"
SEED="${SEED:-42}"
SUFFIX=""; [ "$SEED" != "42" ] && SUFFIX="_s${SEED}"

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

COMPOSER_CKPT="/work/nvme/bfdz/xzhou10/checkpoints/composer_same7b_k${K}${SUFFIX}/composer_epoch4"
QL_PATH=$(ls -t /work/nvme/bfdz/xzhou10/checkpoints/composer_same7b_k${K}${SUFFIX}/query_latents_step*.pt | head -1)
ACTOR="Qwen/Qwen2.5-7B-Instruct"
LIBDIR="/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_same7b_k${K}${SUFFIX}"

mkdir -p "$LIBDIR"
echo "=== Encode 44 skills (same7b, k=${K}) ==="
python -m src.encode_latent_library \
    --composer-ckpt "$COMPOSER_CKPT" --query-latents "$QL_PATH" \
    --actor-path "$ACTOR" --output-dir "$LIBDIR" --variant TRAINED \
    --latents-per-skill ${K} --device cuda


echo "Encode (no bake) k=${K} done."; ls -la "$LIBDIR"
