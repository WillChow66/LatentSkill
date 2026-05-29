#!/bin/bash
#SBATCH --job-name=enc_p2_un
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=logs/enc_p2_un_%j.out
#SBATCH --error=logs/enc_p2_un_%j.err

# Stage 2 Path 2 UNTRAINED: encode with raw Composer + the random ql/proj that
# were FROZEN throughout Path 2 UNTRAINED training. Uses --variant TRAINED so
# encode_latent_library loads the saved (random-init) ql + proj from disk —
# this matches what the Path-2-untrained-actor was conditioned on at SFT.
# Bake into Path-2-untrained-actor (only Actor was trained; composer/ql/proj
# stayed at random init).
# Output: latent_lib_path2_untrained/.

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

COMPOSER_CKPT="Qwen/Qwen2.5-3B-Instruct"
QL_PATH=$(ls -t /work/nvme/bfdz/xzhou10/checkpoints/joint_path2_untrained/query_latents_step*.pt | head -1)
ACTOR_BAKE="/work/nvme/bfdz/xzhou10/checkpoints/joint_path2_untrained/actor_epoch4"
LIBDIR="/work/nvme/bfdz/xzhou10/checkpoints/latent_lib_path2_untrained"
EXPANDED_DIR="${LIBDIR}/actor_expanded_vocab"

mkdir -p "$LIBDIR"

echo "=== Encode 44 skills with Path 2 UNTRAINED Composer (raw + random ql/proj, k=2) ==="
echo "Composer: $COMPOSER_CKPT (raw)"
echo "QL:       $QL_PATH (random init, was frozen during Path 2 untrained SFT)"
python -m src.encode_latent_library \
    --composer-ckpt "$COMPOSER_CKPT" \
    --query-latents "$QL_PATH" \
    --actor-path "$ACTOR_BAKE" \
    --output-dir "$LIBDIR" \
    --variant TRAINED \
    --latents-per-skill 2 \
    --device cuda

echo "=== Bake into Path-2-untrained-actor vocab ==="
python -m src.expand_vocab_with_skills expand \
    --sft-checkpoint "$ACTOR_BAKE" \
    --latent-library "${LIBDIR}/latent_skill_library.pt" \
    --output-dir "$EXPANDED_DIR"

echo "Stage 2 Path 2 UNTRAINED complete."
ls -la "$EXPANDED_DIR"
