#!/bin/bash
#SBATCH --job-name=eval_p2_un
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_p2_un_%j.out
#SBATCH --error=logs/eval_p2_un_%j.err

# Stage-2-only ALFWorld eval for Path 2 UNTRAINED ablation.
# Hypothesis: Path 2 UNTRAINED < TRAINED → joint Composer+Actor SFT gives the
# Composer's discriminative signal value beyond actor-only adaptation.

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

mkdir -p results

MODEL="/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_path2_untrained/actor_expanded_vocab"
LIB="/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_path2_untrained/latent_skill_library.pt"
OUT="results/eval_stage2_path2_untrained_$(date +%Y%m%d_%H%M).json"

python3 -m src.eval_latent_alfworld \
    --model "$MODEL" --latent-library "$LIB" \
    --num-episodes 140 --max-steps 50 \
    --do-sample --temperature 0.4 --top-p 1.0 \
    --history-length 10 --top-k 6 \
    --output "$OUT"

echo "Path 2 UNTRAINED Stage-2 eval done. Result: $OUT"
