#!/bin/bash
#SBATCH --job-name=eval_noise
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_noise_%j.out
#SBATCH --error=logs/eval_noise_%j.err
# Isolate EVAL sampling noise: same fixed lib, 3 evals, different sampling RNG.
# If results swing >5pp, most grid "variance" is eval noise, not training.
set -e
source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1; mkdir -p results
LIB="/work/nvme/bfdz/xzhou10/checkpoints/grid_same7b_k8_s42/lib_epoch4.pt"
for R in 1 2 3; do
  OUT="results/eval_noise_k8s42e4_r${R}_$(date +%Y%m%d_%H%M%S).json"
  python3 -m src.eval_latent_alfworld --model "Qwen/Qwen2.5-7B-Instruct" \
      --latent-library "$LIB" --num-episodes 140 --max-steps 50 \
      --do-sample --temperature 0.4 --top-p 1.0 --history-length 10 --top-k 6 \
      --seed 42 --output "$OUT"
  python3 -c "import json;d=json.load(open('$OUT'));n=len(d['episodes']);print(f'noise run $R: {round(d[\"overall_success_rate\"]*n)}/{n} ({d[\"overall_success_rate\"]*100:.1f}%)')"
done
