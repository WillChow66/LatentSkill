#!/bin/bash
#SBATCH --job-name=eval_rl_newp
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval_rl_newp_%j.out
#SBATCH --error=logs/eval_rl_newp_%j.err
source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill; export PYTHONUNBUFFERED=1; mkdir -p results
OUT="results/eval_skillrl_rl_newparser_$(date +%Y%m%d_%H%M).json"
python3 -m src.eval_text_alfworld --model "/work/nvme/bfdz/xzhou10/checkpoints/skillrl_alfworld_rl_hf" \
    --num-episodes 140 --max-steps 50 --do-sample --temperature 0.4 --top-p 1.0 \
    --history-length 10 --top-k 6 --output "$OUT"
python3 -c "import json;d=json.load(open('$OUT'));n=len(d['episodes']);print(f'RL ckpt NEW parser: {round(d[\"overall_success_rate\"]*n)}/{n} ({d[\"overall_success_rate\"]*100:.1f}%) vs OLD parser 87.9% (anchor must hold)')"
