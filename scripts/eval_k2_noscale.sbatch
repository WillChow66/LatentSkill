#!/bin/bash
#SBATCH --job-name=eval_noscale
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_noscale_%j.out
#SBATCH --error=logs/eval_noscale_%j.err

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1
mkdir -p results

MODEL="/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_clean_k2_noscale/actor_expanded_vocab"
LIB="/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_clean_k2_noscale/latent_skill_library.pt"
OUT="results/eval_stage2_trained_clean_k2_noscale_$(date +%Y%m%d_%H%M).json"

python3 -m src.eval_latent_alfworld \
    --model "$MODEL" --latent-library "$LIB" \
    --num-episodes 140 --max-steps 50 \
    --do-sample --temperature 0.4 --top-p 1.0 \
    --history-length 10 --top-k 6 --output "$OUT"

echo "Eval (k=2 noscale) done: $OUT"
python3 -c "import json; d=json.load(open('$OUT')); n=len(d['episodes']); r=d['overall_success_rate']; print(f'k=2 NOSCALE: {round(r*n)}/{n} ({r*100:.1f}%) — champion k=2 (scale 0.02) is 54/140 (38.6%)')"
