#!/bin/bash
#SBATCH --job-name=verify_vanilla
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/verify_vanilla_%j.out
#SBATCH --error=logs/verify_vanilla_%j.err

# Verify eval on VANILLA 7B + lib.pt == eval on BAKED actor + lib.pt.
# k=8 seed42 baked result was 50.7% (71/140). If vanilla matches within eval
# noise (~2pp), the whole k×epoch×seed grid can eval WITHOUT baking (disk-free).
source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1
mkdir -p results

LIB="/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_same7b_k8/latent_skill_library.pt"
OUT="results/verify_vanilla_k8_s42_$(date +%Y%m%d_%H%M).json"
python3 -m src.eval_latent_alfworld \
    --model "Qwen/Qwen2.5-7B-Instruct" --latent-library "$LIB" \
    --num-episodes 140 --max-steps 50 \
    --do-sample --temperature 0.4 --top-p 1.0 \
    --history-length 10 --top-k 6 --output "$OUT"
python3 -c "import json;d=json.load(open('$OUT'));n=len(d['episodes']);r=d['overall_success_rate'];print(f'VANILLA+lib k=8 s42: {round(r*n)}/{n} ({r*100:.1f}%)  vs BAKED 50.7% (71/140)')"
