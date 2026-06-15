#!/bin/bash
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=03:00:00

# Epoch×seed grid TRAIN: same-model 7B composer (full FT ZeRO-2) + frozen 7B
# actor, --encode-lib-every-epoch → saves lib_epoch1..4.pt (tiny, ~MB each),
# NO 15GB composer ckpt. Output dir: grid_same7b_k${K}_s${SEED}/.
# Submit: sbatch --export=ALL,K=8,SEED=42 -J gtr_k8s42 -o ... -e ... grid_train_kN.sbatch

set -e
: "${K:?K must be set}"; : "${SEED:?SEED must be set}"

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

MODEL7B="Qwen/Qwen2.5-7B-Instruct"
SFT_DATA="/work/nvme/bfdz/xzhou10/huggingface/hub/datasets--Jianwen--SkillRL-SFT-Data/snapshots/bd3996c4e863ac59e6e4ab35549cf3741faf5e4f/alfworld/train-00000-of-00001.parquet"
OUTPUT_DIR="/work/nvme/bfdz/xzhou10/checkpoints/grid_same7b_k${K}_s${SEED}"
DS_CONFIG="/u/xzhou10/latentskill/scripts/ds_config_zero2.json"

mkdir -p "$OUTPUT_DIR"
echo "=== GRID train k=${K} seed=${SEED}: per-epoch in-mem lib, no composer ckpt ==="

accelerate launch \
    --num_processes 4 --use_deepspeed \
    --deepspeed_config_file "$DS_CONFIG" --zero_stage 2 \
    -m src.train_composer_actor_joint \
    --composer-path "$MODEL7B" --actor-path "$MODEL7B" \
    --sft-data "$SFT_DATA" --output-dir "$OUTPUT_DIR" \
    --epochs 4 --composer-lr 1e-5 --latent-lr 5e-3 \
    --max-length 4096 --latents-per-skill ${K} --seed ${SEED} \
    --freeze-actor --encode-lib-every-epoch \
    --save-every 10000 --log-every 50 \
    --wandb-run-name "grid_k${K}_s${SEED}_$(date +%Y%m%d)"

echo "=== train done; per-epoch libs: ==="
ls -la "$OUTPUT_DIR"/lib_epoch*.pt
