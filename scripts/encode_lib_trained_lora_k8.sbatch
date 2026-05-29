#!/bin/bash
#SBATCH --job-name=enc_lora_k8
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=logs/enc_lora_k8_%j.out
#SBATCH --error=logs/enc_lora_k8_%j.err

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

COMPOSER_CKPT="/work/nvme/bfdz/xzhou10/checkpoints/composer_sft_lora_k8/composer_epoch4"
QL_PATH=$(ls -t /work/nvme/bfdz/xzhou10/checkpoints/composer_sft_lora_k8/query_latents_step*.pt | head -1)
ACTOR="Qwen/Qwen2.5-7B-Instruct"
LIBDIR="/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_trained_lora_k8"

mkdir -p "$LIBDIR"
python -m src.encode_latent_library \
    --composer-ckpt "$COMPOSER_CKPT" --query-latents "$QL_PATH" \
    --actor-path "$ACTOR" --output-dir "$LIBDIR" --variant TRAINED \
    --latents-per-skill 8 --device cuda
python -m src.expand_vocab_with_skills expand \
    --sft-checkpoint "$ACTOR" --latent-library "${LIBDIR}/latent_skill_library.pt" \
    --output-dir "${LIBDIR}/actor_expanded_vocab"
ls -la "${LIBDIR}/actor_expanded_vocab"
