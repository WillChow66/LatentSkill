#!/bin/bash
# Apply our modifications to a fresh SkillRL/verl checkout.
# Usage: ./scripts/apply_verl_patches.sh /path/to/SkillRL

set -e

SKILLRL_DIR="${1:-./SkillRL}"

if [ ! -d "$SKILLRL_DIR/verl" ]; then
    echo "ERROR: $SKILLRL_DIR/verl not found. Pass SkillRL clone path as arg."
    echo "Usage: $0 /path/to/SkillRL"
    exit 1
fi

PATCH_DIR="$(dirname "$0")/../verl_patches/verl"

echo "Copying patched verl files to $SKILLRL_DIR/verl/ ..."
cp -v "$PATCH_DIR/utils/vllm_utils.py"                          "$SKILLRL_DIR/verl/utils/vllm_utils.py"
cp -v "$PATCH_DIR/workers/fsdp_workers.py"                      "$SKILLRL_DIR/verl/workers/fsdp_workers.py"
cp -v "$PATCH_DIR/workers/actor/dp_actor.py"                    "$SKILLRL_DIR/verl/workers/actor/dp_actor.py"
cp -v "$PATCH_DIR/workers/critic/dp_critic.py"                  "$SKILLRL_DIR/verl/workers/critic/dp_critic.py"
cp -v "$PATCH_DIR/workers/sharding_manager/fsdp_vllm.py"        "$SKILLRL_DIR/verl/workers/sharding_manager/fsdp_vllm.py"
cp -v "$PATCH_DIR/trainer/ppo/ray_trainer.py"                   "$SKILLRL_DIR/verl/trainer/ppo/ray_trainer.py"

echo ""
echo "Done. Patches applied successfully."
