#!/bin/bash
#SBATCH --job-name=debug_baseline
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/debug_baseline_%j.out
#SBATCH --error=logs/debug_baseline_%j.err
source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1
python3 -m src.debug_baseline_gen
echo "debug_baseline done."
