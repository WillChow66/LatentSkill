#!/bin/bash
#SBATCH --job-name=eval_baseline
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/eval_baseline_%j.out
#SBATCH --error=logs/eval_baseline_%j.err

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

mkdir -p results

python3 -m src.eval_text_alfworld \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --num-episodes 140 \
    --max-steps 50 \
    --top-k 6 \
    --output results/eval_baseline.json

echo "Eval baseline complete."
