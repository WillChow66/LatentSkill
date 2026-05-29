#!/bin/bash
#SBATCH --job-name=eval_skillrl_sft
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_skillrl_sft_%j.out
#SBATCH --error=logs/eval_skillrl_sft_%j.err

# Re-run SkillRL SFT eval with EXPLICIT teammate-aligned flags so all 3 evals
# (TRAINED latent, UNTRAINED latent, SkillRL SFT text) use the IDENTICAL eval
# call signature for paper-grade apples-to-apples. Defaults already matched but
# explicit verification is cleaner.

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

mkdir -p results

SKILLRL_SFT_CKPT="/work/nvme/bdns/xzhou10/huggingface/hub/models--Jianwen--Alfworld-7B-SFT/snapshots/ba9c962eef80a49fc63a94c9728209a36057c671/checkpoint-140"

python3 -m src.eval_text_alfworld \
    --model "$SKILLRL_SFT_CKPT" \
    --num-episodes 140 \
    --max-steps 50 \
    --do-sample \
    --temperature 0.4 \
    --top-p 1.0 \
    --history-length 10 \
    --top-k 6 \
    --output "results/eval_skillrl_sft_v2_$(date +%Y%m%d_%H%M).json"

echo "Eval SkillRL SFT (explicit-flags v2) complete."
