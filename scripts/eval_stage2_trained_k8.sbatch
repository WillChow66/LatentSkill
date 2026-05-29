#!/bin/bash
#SBATCH --job-name=eval_s2_tr_k8
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_s2_tr_k8_%j.out
#SBATCH --error=logs/eval_s2_tr_k8_%j.err

# Stage-2-only eval: TRAINED Composer k=8 (bug version - embed_tokens-frozen).
# Companion to k=2 baseline. Tests if k=8 capacity overcomes the embed-frozen bug.

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

mkdir -p results

MODEL="/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_trained_k8/actor_expanded_vocab"
LIB="/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_trained_k8/latent_skill_library.pt"
OUT="results/eval_stage2_trained_k8_$(date +%Y%m%d_%H%M).json"

python3 -m src.eval_latent_alfworld \
    --model "$MODEL" \
    --latent-library "$LIB" \
    --num-episodes 140 \
    --max-steps 50 \
    --do-sample \
    --temperature 0.4 \
    --top-p 1.0 \
    --history-length 10 \
    --top-k 6 \
    --output "$OUT"

echo "Stage-2 TRAINED k=8 eval done. Result: $OUT"
