#!/bin/bash
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
# Generic: vanilla Qwen2.5-7B + $LIB latent library, NEW robust parser.
# Submit: sbatch --export=ALL,LIB=/path/lib.pt,TAG=name ...
set -e
: "${LIB:?}"; : "${TAG:?}"
source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill; export PYTHONUNBUFFERED=1; mkdir -p results
OUT="results/newparser_${TAG}_$(date +%Y%m%d_%H%M).json"
python3 -m src.eval_latent_alfworld --model "Qwen/Qwen2.5-7B-Instruct" \
    --latent-library "$LIB" --num-episodes 140 --max-steps 50 \
    --do-sample --temperature 0.4 --top-p 1.0 --history-length 10 --top-k 6 --output "$OUT"
python3 -c "import json;d=json.load(open('$OUT'));n=len(d['episodes']);print(f'NEWPARSER ${TAG}: {round(d[\"overall_success_rate\"]*n)}/{n} ({d[\"overall_success_rate\"]*100:.1f}%)')"
