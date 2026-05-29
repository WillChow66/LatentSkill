#!/bin/bash
#SBATCH --job-name=enc_p2_tr
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=logs/enc_p2_tr_%j.out
#SBATCH --error=logs/enc_p2_tr_%j.err

# Stage 2 Path 2 TRAINED: encode 44 skills with joint-trained Composer + ql + proj.
# Bake into Path-2-trained Actor's vocab (NOT vanilla Qwen2.5-7B — this preserves
# the joint-SFT alignment between Composer outputs and Actor latent rows).
# Output: latent_lib_path2_trained/.

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

COMPOSER_CKPT="/work/nvme/bfdz/xzhou10/checkpoints/joint_path2_trained/composer_epoch4"
QL_PATH=$(ls -t /work/nvme/bfdz/xzhou10/checkpoints/joint_path2_trained/query_latents_step*.pt | head -1)
ACTOR_BAKE="/work/nvme/bfdz/xzhou10/checkpoints/joint_path2_trained/actor_epoch4"
LIBDIR="/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_path2_trained"
EXPANDED_DIR="${LIBDIR}/actor_expanded_vocab"

mkdir -p "$LIBDIR"

echo "=== Encode 44 skills with Path 2 TRAINED Composer (k=2) ==="
echo "Composer: $COMPOSER_CKPT"
echo "QL:       $QL_PATH"
python -m src.encode_latent_library \
    --composer-ckpt "$COMPOSER_CKPT" \
    --query-latents "$QL_PATH" \
    --actor-path "$ACTOR_BAKE" \
    --output-dir "$LIBDIR" \
    --variant TRAINED \
    --latents-per-skill 2 \
    --device cuda

echo "=== Bake into Path-2-trained Actor vocab ==="
python -m src.expand_vocab_with_skills expand \
    --sft-checkpoint "$ACTOR_BAKE" \
    --latent-library "${LIBDIR}/latent_skill_library.pt" \
    --output-dir "$EXPANDED_DIR"

echo "Stage 2 Path 2 TRAINED complete."
ls -la "$EXPANDED_DIR"
