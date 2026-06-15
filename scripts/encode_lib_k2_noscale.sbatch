#!/bin/bash
#SBATCH --job-name=enc_noscale
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=logs/enc_noscale_%j.out
#SBATCH --error=logs/enc_noscale_%j.err

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

COMPOSER_CKPT="/work/nvme/bfdz/xzhou10/checkpoints/composer_sft_clean_k2_noscale/composer_epoch4"
QL_PATH=$(ls -t /work/nvme/bfdz/xzhou10/checkpoints/composer_sft_clean_k2_noscale/query_latents_step*.pt | head -1)
ACTOR="Qwen/Qwen2.5-7B-Instruct"
LIBDIR="/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_clean_k2_noscale"

mkdir -p "$LIBDIR"
python -m src.encode_latent_library \
    --composer-ckpt "$COMPOSER_CKPT" --query-latents "$QL_PATH" \
    --actor-path "$ACTOR" --output-dir "$LIBDIR" --variant TRAINED \
    --latents-per-skill 2 --device cuda
python -m src.expand_vocab_with_skills expand \
    --sft-checkpoint "$ACTOR" --latent-library "${LIBDIR}/latent_skill_library.pt" \
    --output-dir "${LIBDIR}/actor_expanded_vocab"
echo "Encode (k=2 noscale) done."
ls -la "${LIBDIR}/actor_expanded_vocab"
