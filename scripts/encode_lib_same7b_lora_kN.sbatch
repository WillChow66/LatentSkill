#!/bin/bash
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:30:00
set -e
: "${K:?}"
source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill; export PYTHONUNBUFFERED=1
CKD="/work/nvme/bfdz/xzhou10/checkpoints/composer_same7b_lora_k${K}"
QL=$(ls -t "$CKD"/query_latents_step*.pt | head -1)
LIBDIR="/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_same7b_lora_k${K}"
mkdir -p "$LIBDIR"
python -m src.encode_latent_library \
    --composer-ckpt "$CKD/composer_epoch4" --query-latents "$QL" \
    --actor-path "Qwen/Qwen2.5-7B-Instruct" --output-dir "$LIBDIR" \
    --variant TRAINED --latents-per-skill ${K} --device cuda
echo "LoRA encode k=${K} done."
