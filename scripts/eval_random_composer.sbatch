#!/bin/bash
#SBATCH --job-name=eval_rand_comp
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_rand_comp_%j.out
#SBATCH --error=logs/eval_rand_comp_%j.err

# Ablation: encode skill library with RANDOM (untrained) query_latents,
# then eval to compare against trained-composer baseline (38/140).
# Same model (global_step_116), same protocol — only query_latents differ.

source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

mkdir -p data/latent_library results

ENCODER="/work/nvme/bdns/xzhou10/checkpoints/latent_skills_fullparam/global_step_116"
RANDOM_QL="data/debug_composer/random_seed42_query_latents.pt"
RANDOM_LIB="data/latent_library/random_seed42_library.pt"

# Step 1: Create random seed=42 query_latents (std=0.02 init)
python3 -c "
import torch
torch.manual_seed(42)
ql = (torch.randn(2, 3584) * 0.02).float()
torch.save({
    'query_latents': ql,
    'latents_per_skill': 2,
    'hidden_size': 3584,
    'model_path': '${ENCODER}',
    'epochs': 0,
    'lr': 0.0,
    'note': 'UNTRAINED random init seed=42 std=0.02',
}, '${RANDOM_QL}')
print(f'Random ql norm={ql.norm().item():.4f}, saved to ${RANDOM_QL}')
"

# Step 2: Encode 44 skills using global_step_116 + random query_latents
python3 -m src.encode_rl_latent_library \
    --model "$ENCODER" \
    --query-latents "$RANDOM_QL" \
    --output "$RANDOM_LIB" \
    --skill-token-map "/work/nvme/bdns/xzhou10/checkpoints/latent_skills_token_v2/skill_token_map.json" \
    --latents-per-skill 2

# Step 3: Eval with global_step_116 + random library
python3 -m src.eval_latent_alfworld \
    --model "$ENCODER" \
    --latent-library "$RANDOM_LIB" \
    --num-episodes 140 \
    --max-steps 50 \
    --output results/eval_random_composer.json

echo "Random composer eval complete."
