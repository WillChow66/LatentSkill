#!/bin/bash
#SBATCH --job-name=composer
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/composer_%j.out
#SBATCH --error=logs/composer_%j.err

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

# Use SkillRL's SFT checkpoint as the base model (agent model = encoder)
MODEL_PATH="/work/nvme/bdns/xzhou10/checkpoints/latent_skills_fullparam/global_step_116"
SFT_DATA="/work/nvme/bdns/xzhou10/huggingface/hub/datasets--Jianwen--SkillRL-SFT-Data/snapshots/b09e9ebebfd858379196fa2fb508f8706d251292/alfworld/train-00000-of-00001.parquet"
SKILLS_JSON="SkillRL/memory_data/alfworld/claude_style_skills.json"

python3 -m src.train_composer \
    --model-path $MODEL_PATH \
    --sft-data $SFT_DATA \
    --skills-json $SKILLS_JSON \
    --output-dir data/trained_composer \
    --latents-per-skill 2 \
    --epochs 10 \
    --lr 5e-3 \
    --max-length 4096 \
    --seed 42
