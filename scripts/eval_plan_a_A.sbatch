#!/bin/bash
#SBATCH --job-name=eval_planA_A
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_planA_A_%j.out
#SBATCH --error=logs/eval_planA_A_%j.err

# Plan A Group A eval: joint-trained model + joint-trained query_latents

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

mkdir -p data/latent_library results

MODEL="/work/nvme/bfdz/xzhou10/checkpoints/joint_sft_A/model_epoch4"
QL="/work/nvme/bfdz/xzhou10/checkpoints/joint_sft_A/query_latents_step7484.pt"
LIB="data/latent_library/plan_a_A_library.pt"

# Step 1: encode 44 skills with A's model + A's trained query_latents
python3 -m src.encode_plan_a_library \
    --model "$MODEL" \
    --query-latents "$QL" \
    --output "$LIB" \
    --latents-per-skill 2

# Step 2: eval (sample, temp=0.4, history=10 — matching teammate protocol)
python3 -m src.eval_latent_alfworld \
    --model "$MODEL" \
    --latent-library "$LIB" \
    --num-episodes 140 \
    --max-steps 50 \
    --output results/eval_plan_a_A.json

echo "Plan A Group A eval done."
