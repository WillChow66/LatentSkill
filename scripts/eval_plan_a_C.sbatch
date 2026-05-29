#!/bin/bash
#SBATCH --job-name=eval_planA_C
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_planA_C_%j.out
#SBATCH --error=logs/eval_planA_C_%j.err

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1
mkdir -p data/latent_library results

MODEL="/work/nvme/bfdz/xzhou10/checkpoints/joint_sft_C_frozen_pretrained/model_epoch4"
QL="/work/nvme/bfdz/xzhou10/checkpoints/joint_sft_C_frozen_pretrained/query_latents_step7484.pt"
LIB="data/latent_library/plan_a_C_library.pt"

python3 -m src.encode_plan_a_library --model "$MODEL" --query-latents "$QL" --output "$LIB" --latents-per-skill 2

python3 -m src.eval_latent_alfworld --model "$MODEL" --latent-library "$LIB" --num-episodes 140 --max-steps 50 --output results/eval_plan_a_C.json
echo "Plan A Group C eval done."
