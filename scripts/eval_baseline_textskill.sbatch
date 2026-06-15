#!/bin/bash
#SBATCH --job-name=eval_textskill
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/eval_textskill_%j.out
#SBATCH --error=logs/eval_textskill_%j.err

# Clean 3-way baseline (2/3): TEXT skills in-context on vanilla Qwen2.5-7B.
# Same protocol as latent evals: 140 ep, T=0.4, top_p=1.0, history=10.
# Re-confirms the (suspicious) 0% from eval_baseline.json with a clean dated record.

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1
mkdir -p results

OUT="results/eval_baseline_textskill_$(date +%Y%m%d_%H%M).json"
python3 -m src.eval_text_alfworld \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --num-episodes 140 --max-steps 50 \
    --do-sample --temperature 0.4 --top-p 1.0 \
    --history-length 10 --top-k 6 \
    --output "$OUT"

echo "Text-skill baseline done: $OUT"
python3 -c "import json; d=json.load(open('$OUT')); n=len(d['episodes']); r=d['overall_success_rate']; print(f'TEXT-SKILL: {round(r*n)}/{n} ({r*100:.1f}%)  mode={d[\"mode\"]}  tok={d[\"avg_tokens\"]:.0f}')"
