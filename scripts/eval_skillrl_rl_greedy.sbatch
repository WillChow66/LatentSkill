#!/bin/bash
#SBATCH --job-name=eval_rl_greedy
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval_rl_greedy_%j.out
#SBATCH --error=logs/eval_rl_greedy_%j.err

# Determinism test: SkillRL RL ckpt with GREEDY decoding (T=0, do_sample=False).
# Compare vs T=0.4 (existing 87.9%) and paper (89.9%). Fully deterministic
# (fixed seed=42 + greedy) -> reproducible eval protocol candidate.

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1
mkdir -p results

SKILLRL_RL_CKPT="/work/nvme/bfdz/xzhou10/checkpoints/skillrl_alfworld_rl_hf"
OUT="results/eval_skillrl_rl_greedy_$(date +%Y%m%d_%H%M).json"

python3 -m src.eval_text_alfworld \
    --model "$SKILLRL_RL_CKPT" \
    --num-episodes 140 --max-steps 50 \
    --no-do-sample \
    --history-length 10 --top-k 6 \
    --output "$OUT"

echo "SkillRL RL greedy (T=0) eval done: $OUT"
python3 -c "import json; d=json.load(open('$OUT')); n=len(d['episodes']); r=d['overall_success_rate']; print(f'T=0 GREEDY: {round(r*n)}/{n} ({r*100:.1f}%)  vs T=0.4=87.9%  vs paper=89.9%  tok={d[\"avg_tokens\"]:.0f}')"
