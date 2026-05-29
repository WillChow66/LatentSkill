#!/bin/bash
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00

# K-sweep eval template. Reads K from env. Output: results/eval_stage2_trained_clean_k${K}_*.json.
# Compare against current champion: Full P1 TRAINED k=2 = 54/140 (38.6%).

set -e
: "${K:?K env var must be set}"

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

mkdir -p results

MODEL="/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_trained_clean_k${K}/actor_expanded_vocab"
LIB="/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_trained_clean_k${K}/latent_skill_library.pt"
OUT="results/eval_stage2_trained_clean_k${K}_$(date +%Y%m%d_%H%M).json"

echo "=== Stage-2 eval k=${K} (140 ep, do_sample=True T=0.4) ==="
python3 -m src.eval_latent_alfworld \
    --model "$MODEL" --latent-library "$LIB" \
    --num-episodes 140 --max-steps 50 \
    --do-sample --temperature 0.4 --top-p 1.0 \
    --history-length 10 --top-k 6 \
    --output "$OUT"

echo "Eval k=${K} done. Result: $OUT"
python3 -c "import json; d=json.load(open('$OUT')); n=len(d['episodes']); r=d['overall_success_rate']; print(f'k=${K}: {round(r*n)}/{n} ({r*100:.1f}%) — champion k=2 is 54/140 (38.6%)')"
