#!/bin/bash
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
# UNTRAINED same-7B control: encode 44 skills with RAW Qwen2.5-7B composer
# (random ql, Identity proj since D==D), eval on vanilla 7B with new parser.
# Apples-to-apples control for the trained same-7B k-table.
set -e
: "${K:?}"
source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill; export PYTHONUNBUFFERED=1; mkdir -p results
LIBDIR="/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_untrained_same7b_k${K}"
mkdir -p "$LIBDIR"
python -m src.encode_latent_library \
    --composer-ckpt "Qwen/Qwen2.5-7B-Instruct" --actor-path "Qwen/Qwen2.5-7B-Instruct" \
    --output-dir "$LIBDIR" --variant UNTRAINED --latents-per-skill ${K} --seed 42 --device cuda
OUT="results/newparser_untrained_same7b_k${K}_$(date +%Y%m%d_%H%M).json"
python3 -m src.eval_latent_alfworld --model "Qwen/Qwen2.5-7B-Instruct" \
    --latent-library "${LIBDIR}/latent_skill_library.pt" --num-episodes 140 --max-steps 50 \
    --do-sample --temperature 0.4 --top-p 1.0 --history-length 10 --top-k 6 --output "$OUT"
python3 -c "import json;d=json.load(open('$OUT'));n=len(d['episodes']);print(f'UNTRAINED same7b k=${K}: {round(d[\"overall_success_rate\"]*n)}/{n} ({d[\"overall_success_rate\"]*100:.1f}%)')"
