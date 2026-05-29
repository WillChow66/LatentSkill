#!/usr/bin/env bash
set -euo pipefail

cd /home/nvidia/Jiashu/SkillRL

LOG_DIR="/home/nvidia/Jiashu/SkillRL/logs/nohup_log/step0_eval_local"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_full_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

export PYTHONPATH="/home/nvidia/Jiashu/SkillRL:${PYTHONPATH:-}"
export ALFWORLD_DATA="/home/nvidia/Jiashu/SkillRL/alfworld_data"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

MODEL_PATH="${MODEL_PATH:-/mnt/agent-project/ckpts/Alfworld-7B-RL}"
MERGED_MODEL_CACHE_DIR="${MERGED_MODEL_CACHE_DIR:-}"
EVAL_DATASET="${EVAL_DATASET:-eval_in_distribution}"
TOTAL_GAMES="${TOTAL_GAMES:-20}"
BATCH_SIZE="${BATCH_SIZE:-20}"
MAX_STEPS="${MAX_STEPS:-50}"
TEMPERATURE="${TEMPERATURE:-0.4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
RUN_PREFIX="${RUN_PREFIX:-qwen_local_skillbank_eval_id}"
SKILLS_JSON_PATH="${SKILLS_JSON_PATH:-/home/nvidia/Jiashu/SkillRL/memory_data/alfworld/claude_style_skills.json}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-template}"
TOP_K="${TOP_K:-6}"
TASK_SPECIFIC_TOP_K="${TASK_SPECIFIC_TOP_K:-}"
UPDATE_THRESHOLD="${UPDATE_THRESHOLD:-0.4}"
MAX_NEW_SKILLS="${MAX_NEW_SKILLS:-3}"

echo "Model path: ${MODEL_PATH}"
echo "Merged model cache dir: ${MERGED_MODEL_CACHE_DIR:-<per-run temporary>}"
echo "Eval dataset: ${EVAL_DATASET}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Total games: ${TOTAL_GAMES}"
echo "Batch size: ${BATCH_SIZE}"

offset=0
batch_idx=0

while [ "${offset}" -lt "${TOTAL_GAMES}" ]; do
  current_num_games="${BATCH_SIZE}"
  remaining=$((TOTAL_GAMES - offset))
  if [ "${remaining}" -lt "${current_num_games}" ]; then
    current_num_games="${remaining}"
  fi

  run_name="${RUN_PREFIX}_offset${offset}_n${current_num_games}"

  echo "========== Batch ${batch_idx} | offset=${offset} num_games=${current_num_games} =========="

  cmd=(
    python -m examples.prompt_agent.step0_eval_local
    --model-path "${MODEL_PATH}"
    --env-num "${current_num_games}"
    --game-offset "${offset}"
    --num-games "${current_num_games}"
    --max-steps "${MAX_STEPS}"
    --eval-dataset "${EVAL_DATASET}"
    --temperature "${TEMPERATURE}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --use-skills-only-memory
    --skills-json-path "${SKILLS_JSON_PATH}"
    --retrieval-mode "${RETRIEVAL_MODE}"
    --top-k "${TOP_K}"
    --enable-dynamic-update
    --update-threshold "${UPDATE_THRESHOLD}"
    --max-new-skills "${MAX_NEW_SKILLS}"
    --run-name "${run_name}"
  )

  if [ -n "${TASK_SPECIFIC_TOP_K}" ]; then
    cmd+=(--task-specific-top-k "${TASK_SPECIFIC_TOP_K}")
  fi

  if [ -n "${MERGED_MODEL_CACHE_DIR}" ]; then
    cmd+=(--merged-model-cache-dir "${MERGED_MODEL_CACHE_DIR}")
  fi

  "${cmd[@]}"

  offset=$((offset + current_num_games))
  batch_idx=$((batch_idx + 1))
done

echo "All batches completed."

# nohup bash /home/nvidia/Jiashu/SkillRL/examples/prompt_agent/step0_eval_local.sh > /home/nvidia/Jiashu/SkillRL/logs/nohup_log/step0_eval_local/run_full.log 2>&1 &
