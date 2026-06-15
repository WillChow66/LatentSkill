#!/bin/bash
#SBATCH --job-name=txt_full
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/txt_full_%j.out
#SBATCH --error=logs/txt_full_%j.err
# FULL-text anchor: top_k=12 => ALL 12 general + ALL task-specific + mistakes
# (~17 skills) = the exact same skill set the latent path injects. The truly
# apples-to-apples text<->latent cell (the top-6 anchor under-supplies text).
set -e
source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill; export PYTHONUNBUFFERED=1; mkdir -p results
OUT="results/newparser_textskill_full_$(date +%Y%m%d_%H%M).json"
python3 -m src.eval_text_alfworld --model "Qwen/Qwen2.5-7B-Instruct" \
    --num-episodes 140 --max-steps 50 --do-sample --temperature 0.4 --top-p 1.0 \
    --history-length 10 --top-k 12 --output "$OUT"
python3 -c "import json;d=json.load(open('$OUT'));n=len(d['episodes']);print(f'TEXT-FULL(top12, ~17 skills): {round(d[\"overall_success_rate\"]*n)}/{n} ({d[\"overall_success_rate\"]*100:.1f}%) tok={d[\"avg_tokens\"]:.0f}')"
