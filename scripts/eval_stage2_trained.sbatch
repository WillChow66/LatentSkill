#!/bin/bash
#SBATCH --job-name=eval_s2_tr
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_s2_tr_%j.out
#SBATCH --error=logs/eval_s2_tr_%j.err

# Stage-2-only eval (TRAINED variant): pure latent quality test, NO RL adapt.
# Actor = vocab-expanded with TRAINED-Composer-encoded latents (44 SKILL tokens).
# This is the paired eval to UNTRAINED — together they answer:
#   "Does Composer SFT actually help the latent representations be useful?"
# Same eval protocol as UNTRAINED: sample, temp=0.4, top_p=1.0, history=10,
# max_steps=50, 140 episodes valid_in_distribution. Matches teammate (123/140 baseline).

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

mkdir -p results

MODEL="/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_trained/actor_expanded_vocab"
LIB="/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_trained/latent_skill_library.pt"
OUT="results/eval_stage2_trained_$(date +%Y%m%d_%H%M).json"

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

echo "Stage-2 TRAINED eval done. Result: $OUT"
