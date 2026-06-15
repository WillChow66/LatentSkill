#!/bin/bash
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00

# Epoch×seed grid EVAL: vanilla 7B + lib_epoch{EPOCH}.pt (verified equivalent to
# baked actor, 2.1pp within noise). Disk-free. Output: results/grid_k${K}_s${SEED}_e${EPOCH}_*.json
# Submit: sbatch --export=ALL,K=8,SEED=42,EPOCH=2 -J gev_k8s42e2 -o ... -e ... grid_eval_kN.sbatch

set -e
: "${K:?K must be set}"; : "${SEED:?SEED must be set}"; : "${EPOCH:?EPOCH must be set}"

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1
mkdir -p results

LIB="/work/nvme/bfdz/xzhou10/checkpoints/grid_same7b_k${K}_s${SEED}/lib_epoch${EPOCH}.pt"
OUT="results/grid_k${K}_s${SEED}_e${EPOCH}_$(date +%Y%m%d_%H%M).json"

python3 -m src.eval_latent_alfworld \
    --model "Qwen/Qwen2.5-7B-Instruct" --latent-library "$LIB" \
    --num-episodes 140 --max-steps 50 \
    --do-sample --temperature 0.4 --top-p 1.0 \
    --history-length 10 --top-k 6 --output "$OUT"

python3 -c "import json;d=json.load(open('$OUT'));n=len(d['episodes']);r=d['overall_success_rate'];print(f'GRID k=${K} s=${SEED} e=${EPOCH}: {round(r*n)}/{n} ({r*100:.1f}%)')"
