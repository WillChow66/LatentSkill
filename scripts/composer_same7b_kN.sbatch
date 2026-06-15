#!/bin/bash
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=04:00:00

# Same-model k-sweep: Composer = Actor = Qwen2.5-7B-Instruct.
#   - Composer 7B FULL FT (ZeRO-2 shards optimizer states; fp32 master included)
#   - Actor 7B FROZEN (--freeze-actor; ckpt not saved — weights stay vanilla)
#   - latent_proj auto-Identity (D 3584 == 3584): NO cross-model projection
# Tests whether k-monotonicity appears once the cross-model gap is removed
# (full FT per user decision; LoRA arm deferred). Submit:
#   sbatch --export=ALL,K=2 --job-name=same7b_k2 \
#          -o logs/same7b_k2_%j.out -e logs/same7b_k2_%j.err \
#          scripts/composer_same7b_kN.sbatch

set -e
: "${K:?K env var must be set, e.g. --export=ALL,K=2}"
# Optional SEED env (default 42 = legacy runs, no dir suffix). SEED!=42 runs get
# a _s${SEED} suffix for multi-seed variance studies.
SEED="${SEED:-42}"
SUFFIX=""; [ "$SEED" != "42" ] && SUFFIX="_s${SEED}"

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

MODEL7B="Qwen/Qwen2.5-7B-Instruct"
SFT_DATA="/work/nvme/bfdz/xzhou10/huggingface/hub/datasets--Jianwen--SkillRL-SFT-Data/snapshots/bd3996c4e863ac59e6e4ab35549cf3741faf5e4f/alfworld/train-00000-of-00001.parquet"
OUTPUT_DIR="/work/nvme/bfdz/xzhou10/checkpoints/composer_same7b_k${K}${SUFFIX}"
DS_CONFIG="/u/xzhou10/latentskill/scripts/ds_config_zero2.json"

mkdir -p "$OUTPUT_DIR"
echo "=== Same-model SFT k=${K} seed=${SEED}: 7B composer (full FT, ZeRO-2) + 7B actor (frozen) ==="

accelerate launch \
    --num_processes 4 \
    --use_deepspeed \
    --deepspeed_config_file "$DS_CONFIG" \
    --zero_stage 2 \
    -m src.train_composer_actor_joint \
    --composer-path "$MODEL7B" \
    --actor-path "$MODEL7B" \
    --sft-data "$SFT_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 4 \
    --composer-lr 1e-5 \
    --latent-lr 5e-3 \
    --max-length 4096 \
    --latents-per-skill ${K} \
    --seed ${SEED} \
    --freeze-actor \
    --save-every 10000 \
    --log-every 20 \
    --wandb-run-name "composer_same7b_k${K}${SUFFIX}_$(date +%Y%m%d)"

echo "Same-model SFT k=${K} done."
ls -la "$OUTPUT_DIR"
