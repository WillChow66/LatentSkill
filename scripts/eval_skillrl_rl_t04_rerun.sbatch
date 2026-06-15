#!/bin/bash
#SBATCH --job-name=eval_rl_t04
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval_rl_t04_%j.out
#SBATCH --error=logs/eval_rl_t04_%j.err

# Variance test: SkillRL RL ckpt at T=0.4 (sampling), FRESH second sample.
# Compare vs the existing eval_skillrl_rl.json (87.9%) to quantify how much
# the same ckpt+protocol fluctuates run-to-run = the sampling noise floor.

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1
mkdir -p results

SKILLRL_RL_CKPT="/work/nvme/bfdz/xzhou10/checkpoints/skillrl_alfworld_rl_hf"
OUT="results/eval_skillrl_rl_t04_rerun_$(date +%Y%m%d_%H%M).json"

python3 -m src.eval_text_alfworld \
    --model "$SKILLRL_RL_CKPT" \
    --num-episodes 140 --max-steps 50 \
    --do-sample --temperature 0.4 --top-p 1.0 \
    --history-length 10 --top-k 6 \
    --output "$OUT"

echo "SkillRL RL T=0.4 rerun done: $OUT"
python3 -c "import json; d=json.load(open('$OUT')); n=len(d['episodes']); r=d['overall_success_rate']; print(f'T=0.4 rerun: {round(r*n)}/{n} ({r*100:.1f}%)  vs first T=0.4=87.9%  -> diff = sampling noise')"
