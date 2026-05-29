#!/bin/bash
# Submit K-sweep chain: K=1, 3, 4 (each chain = SFT → encode → eval).
# Current champion: k=2 = 54/140 (38.6%). Cleanup decision is MANUAL after
# all three eval JSONs are in (review then rm losers).
#
# Usage: bash scripts/submit_k_sweep.sh

set -e
cd /u/xzhou10/latentskill

declare -A IDS
for K in 1 3 4; do
    SFT=$(sbatch --parsable \
        --export=ALL,K=$K \
        --job-name="composer_k${K}" \
        -o "logs/composer_k${K}_%j.out" -e "logs/composer_k${K}_%j.err" \
        scripts/composer_sft_clean_kN.sbatch)

    ENC=$(sbatch --parsable --dependency=afterok:$SFT \
        --export=ALL,K=$K \
        --job-name="enc_clean_k${K}" \
        -o "logs/enc_clean_k${K}_%j.out" -e "logs/enc_clean_k${K}_%j.err" \
        scripts/encode_lib_trained_clean_kN.sbatch)

    EVAL=$(sbatch --parsable --dependency=afterok:$ENC \
        --export=ALL,K=$K \
        --job-name="eval_clean_k${K}" \
        -o "logs/eval_clean_k${K}_%j.out" -e "logs/eval_clean_k${K}_%j.err" \
        scripts/eval_stage2_trained_clean_kN.sbatch)

    IDS[K${K}]="SFT=$SFT ENC=$ENC EVAL=$EVAL"
    echo "K=${K}: SFT=$SFT → ENC=$ENC → EVAL=$EVAL"
done

echo
echo "=== Submitted 9 jobs (3 K values × 3 stages each) ==="
for K in 1 3 4; do
    echo "  K=${K}: ${IDS[K${K}]}"
done
