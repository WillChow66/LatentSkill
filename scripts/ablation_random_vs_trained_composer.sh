#!/bin/bash
# Ablation: random query_latents vs Stage-1 trained query_latents
# Uses global_step_116 as the encoder (matches existing trained library)
# After this, run eval scripts to compare.

set -e

source /u/xzhou10/latentskill/scripts/setup_env.sh

DATA_DIR="/u/xzhou10/latentskill/data/debug_composer"
RANDOM_QL="${DATA_DIR}/random_seed42_query_latents.pt"
RANDOM_LIB="/u/xzhou10/latentskill/data/latent_library/random_seed42_library.pt"

# Step 1: Create random seed=42 query_latents in the format expected by SkillComposer.load_pretrained
python3 -c "
import torch
torch.manual_seed(42)
ql = (torch.randn(2, 3584) * 0.02).float()
torch.save({
    'query_latents': ql,
    'latents_per_skill': 2,
    'hidden_size': 3584,
    'model_path': '/work/nvme/bdns/xzhou10/checkpoints/latent_skills_fullparam/global_step_116',
    'epochs': 0,
    'lr': 0.0,
    'note': 'untrained, seed=42 random init (std=0.02)',
}, '${RANDOM_QL}')
print(f'Saved random query_latents (norm={ql.norm().item():.4f}) to ${RANDOM_QL}')
"

# Step 2: Encode skill library using global_step_116 + random query_latents
mkdir -p $(dirname "$RANDOM_LIB")
cd /u/xzhou10/latentskill
python3 -m src.encode_rl_latent_library \
    --model "/work/nvme/bdns/xzhou10/checkpoints/latent_skills_fullparam/global_step_116" \
    --query-latents "$RANDOM_QL" \
    --output "$RANDOM_LIB" \
    --skill-token-map "/work/nvme/bdns/xzhou10/checkpoints/latent_skills_token_v2/skill_token_map.json" \
    --latents-per-skill 2

ls -la "$RANDOM_LIB"
echo "Random library encoded. Compare via eval scripts."
