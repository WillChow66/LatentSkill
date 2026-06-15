#!/bin/bash
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00

# Same-model k-sweep eval. Monotonicity readout: compare across K in {1,2,4,8}
# against 3B-composer curve (32.1 / 38.6 / 20.0 / 32.9 for k=1/2/4/8).

set -e
: "${K:?K env var must be set}"
SEED="${SEED:-42}"
SUFFIX=""; [ "$SEED" != "42" ] && SUFFIX="_s${SEED}"

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1
mkdir -p results

CKPTROOT="/work/nvme/bfdz/xzhou10/checkpoints"
MODEL="${CKPTROOT}/latent_lib_same7b_k${K}${SUFFIX}/actor_expanded_vocab"
LIB="${CKPTROOT}/latent_lib_same7b_k${K}${SUFFIX}/latent_skill_library.pt"
OUT="results/eval_stage2_same7b_k${K}${SUFFIX}_$(date +%Y%m%d_%H%M).json"

python3 -m src.eval_latent_alfworld \
    --model "$MODEL" --latent-library "$LIB" \
    --num-episodes 140 --max-steps 50 \
    --do-sample --temperature 0.4 --top-p 1.0 \
    --history-length 10 --top-k 6 --output "$OUT"

echo "same7b k=${K} seed=${SEED} eval done: $OUT"
python3 -c "import json; d=json.load(open('$OUT')); n=len(d['episodes']); r=d['overall_success_rate']; print(f'same7b k=${K} seed=${SEED}: {round(r*n)}/{n} ({r*100:.1f}%)')"

# Multi-seed runs (SEED != 42) auto-churn their heavy artifacts after a
# successful eval (JSON written + parsed above, set -e guarantees we only get
# here on success). lib.pt and the eval JSON are preserved.
if [ -n "$SUFFIX" ]; then
    rm -rf "${CKPTROOT}/composer_same7b_k${K}${SUFFIX}" \
           "${CKPTROOT}/latent_lib_same7b_k${K}${SUFFIX}/actor_expanded_vocab"
    echo "Auto-churned seed-run artifacts for k=${K}${SUFFIX} (kept lib.pt + JSON)."
fi
