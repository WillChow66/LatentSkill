#!/bin/bash
#SBATCH --job-name=verify_encode_equiv
#SBATCH --account=bfdz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --output=logs/verify_encode_equiv_%j.out
#SBATCH --error=logs/verify_encode_equiv_%j.err

# Decisive test: does the in-memory per-epoch encode (--encode-lib-every-epoch)
# produce the SAME library as the trusted separate encode_latent_library.py from
# a saved ckpt of the same state? If yes, the new code path is correct and the
# grid's variance is training nondeterminism, not a code artifact.
# Trains a tiny smoke with BOTH flags (saves composer ckpt AND in-mem lib), then
# separately encodes the saved ckpt and diffs the two libraries.
set -e
source /u/xzhou10/latentskill/scripts/setup_env.sh
cd /u/xzhou10/latentskill
export PYTHONUNBUFFERED=1

MODEL7B="Qwen/Qwen2.5-7B-Instruct"
SFT_DATA="/work/nvme/bfdz/xzhou10/huggingface/hub/datasets--Jianwen--SkillRL-SFT-Data/snapshots/bd3996c4e863ac59e6e4ab35549cf3741faf5e4f/alfworld/train-00000-of-00001.parquet"
OUT="/work/nvme/bfdz/xzhou10/checkpoints/verify_encode_equiv"
DS_CONFIG="/u/xzhou10/latentskill/scripts/ds_config_zero2.json"
rm -rf "$OUT"; mkdir -p "$OUT"

echo "=== Train smoke (2 epoch x 80 samples), BOTH flags on ==="
accelerate launch --num_processes 4 --use_deepspeed \
    --deepspeed_config_file "$DS_CONFIG" --zero_stage 2 \
    -m src.train_composer_actor_joint \
    --composer-path "$MODEL7B" --actor-path "$MODEL7B" \
    --sft-data "$SFT_DATA" --output-dir "$OUT" \
    --epochs 2 --max-samples 80 --composer-lr 1e-5 --latent-lr 5e-3 \
    --max-length 4096 --latents-per-skill 8 --seed 42 \
    --freeze-actor --encode-lib-every-epoch --save-composer-every-epoch \
    --log-every 10 --no-wandb

echo "=== Separately encode the saved epoch2 ckpt (trusted path) ==="
QL=$(ls -t "$OUT"/query_latents_step*.pt | head -1)
python -m src.encode_latent_library \
    --composer-ckpt "$OUT/composer_epoch2" --query-latents "$QL" \
    --actor-path "$MODEL7B" --output-dir "$OUT/separate_encode" \
    --variant TRAINED --latents-per-skill 8 --device cuda

echo "=== Diff in-memory lib_epoch2.pt vs separate encode ==="
python3 - <<'PYEOF'
import torch
a = torch.load("/work/nvme/bfdz/xzhou10/checkpoints/verify_encode_equiv/lib_epoch2.pt", map_location="cpu", weights_only=False)["latent_library"]
b = torch.load("/work/nvme/bfdz/xzhou10/checkpoints/verify_encode_equiv/separate_encode/latent_skill_library.pt", map_location="cpu", weights_only=False)["latent_library"]
keys = sorted(set(a) & set(b))
import torch.nn.functional as F
maxdiff = 0.0; mincos = 1.0
for k in keys:
    va, vb = a[k].float(), b[k].float()
    maxdiff = max(maxdiff, (va-vb).abs().max().item())
    mincos = min(mincos, F.cosine_similarity(va.flatten(), vb.flatten(), dim=0).item())
print(f"compared {len(keys)} skills | max |abs diff|={maxdiff:.4f} | min cossim={mincos:.6f}")
print("VERDICT:", "IDENTICAL (in-mem encode is correct)" if maxdiff < 0.05 and mincos > 0.999
      else "DIFFERENT (new code path changes the library!)")
PYEOF
rm -rf "$OUT"
echo "cleaned up smoke ckpts."
