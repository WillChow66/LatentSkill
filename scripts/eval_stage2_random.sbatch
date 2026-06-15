#!/bin/bash
#SBATCH --job-name=eval_random
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval_random_%j.out
#SBATCH --error=logs/eval_random_%j.err

# Eval the garbage-latent control. Same 140-ep T=0.4 protocol as all others.
# Interpretation grid: RANDOM ≈ no-skill (~2%) => UNTRAINED 23.6% is real info;
# RANDOM ≈ 23% => scaffold/any-vector effect, UNTRAINED interpretation wrong.

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1
mkdir -p results

MODEL="/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_random/actor_expanded_vocab"
LIB="/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_random/latent_skill_library.pt"
OUT="results/eval_stage2_random_$(date +%Y%m%d_%H%M).json"

python3 -m src.eval_latent_alfworld \
    --model "$MODEL" --latent-library "$LIB" \
    --num-episodes 140 --max-steps 50 \
    --do-sample --temperature 0.4 --top-p 1.0 \
    --history-length 10 --top-k 6 --output "$OUT"

echo "Garbage-latent eval done: $OUT"
python3 -c "import json; d=json.load(open('$OUT')); n=len(d['episodes']); r=d['overall_success_rate']; print(f'RANDOM latents: {round(r*n)}/{n} ({r*100:.1f}%)  [UNTRAINED=23.6%, no-skill~2%]')"
