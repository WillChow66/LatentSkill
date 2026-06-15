#!/bin/bash
#SBATCH --job-name=debug_k1
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --output=logs/debug_k1_%j.out
#SBATCH --error=logs/debug_k1_%j.err

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1
python3 -m src.debug_k1_generations --episodes 2 --steps 6
echo "debug_k1 done."
