#!/usr/bin/env bash
# Smoke-test EU-Guard eval harness on JUWELS Booster (devel queue, fast turnaround).
#
# Submit from login node (from EU_GUARD_ROOT):
#   mkdir -p eval/logs
#   sbatch eval_harness/scripts/hpc/sbatch_smoke_juwels.sh
#
# Prefetch models on the login node into the same HF_CACHE this job uses
# (see USER_SHARED/hf_cache below) before submitting — compute nodes have
# no internet.
#
# Override paths/accounts via environment variables before sbatch.

#SBATCH --account=laionize
#SBATCH --partition=develbooster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
#SBATCH --job-name=eu-guard-smoke
#SBATCH --output=eval/logs/slurm-%x-%j.out
#SBATCH --error=eval/logs/slurm-%x-%j.err

set -euo pipefail
ulimit -c 0

PROJECT_ACCOUNT="${PROJECT_ACCOUNT:-laionize}"
MACHINE="${MACHINE:-juwels}"
NAME_ENV="${NAME_ENV:-eu_guard_eval}"
EU_GUARD_ROOT="${EU_GUARD_ROOT:-/p/project1/${PROJECT_ACCOUNT}/${USER}/eu-guard}"

USER_SHARED="/p/scratch/${PROJECT_ACCOUNT}/${USER}_${MACHINE}_shared"
SHARED_MAMBA_ENV="${USER_SHARED}/mamba/base_${NAME_ENV}"
SHARED_ENV="${USER_SHARED}/envs/mamba_${NAME_ENV}"
HF_CACHE="${USER_SHARED}/hf_cache"

MODEL="${VLLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
JUDGE_MODEL="${VLLM_JUDGE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_PORT="${VLLM_PORT:-8000}"
JUDGE_PORT="${VLLM_JUDGE_PORT:-8001}"
SAMPLE_SIZE="${SMOKE_SAMPLE_SIZE:-20}"

module --force purge
source "${SHARED_MAMBA_ENV}/bin/activate" "${SHARED_ENV}"

# Compute nodes have no internet — use prefetched scratch cache only.
export HF_HOME="${HF_CACHE}"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_PATH="${HF_CACHE}"
export VLLM_JUDGE_CACHE_PATH="${HF_CACHE}"

cd "${EU_GUARD_ROOT}"
mkdir -p eval/logs

wait_for_server() {
  local port="$1"
  local label="$2"
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      echo "${label} ready on port ${port}"
      return 0
    fi
    sleep 5
  done
  echo "Timed out waiting for ${label} on port ${port}" >&2
  return 1
}

CUDA_VISIBLE_DEVICES=0 \
  VLLM_MODEL="${MODEL}" VLLM_PORT="${MODEL_PORT}" VLLM_TP_SIZE=1 \
  VLLM_CACHE_PATH="${HF_CACHE}" \
  bash eval_harness/scripts/start_vllm_model_server.sh &
MODEL_PID=$!

wait_for_server "${MODEL_PORT}" "model server"

CUDA_VISIBLE_DEVICES=1 \
  VLLM_JUDGE_MODEL="${JUDGE_MODEL}" VLLM_JUDGE_PORT="${JUDGE_PORT}" VLLM_JUDGE_TP_SIZE=1 \
  VLLM_JUDGE_CACHE_PATH="${HF_CACHE}" \
  bash eval_harness/scripts/start_vllm_judge_server.sh &
JUDGE_PID=$!

wait_for_server "${JUDGE_PORT}" "judge server"

python3 -m eval_harness.cli plan \
  --dataset-path eval_harness/data/EU_alert_working_copy/test/test.jsonl \
  --output-dir eval \
  --model-name smoke-model \
  --backend vllm \
  --model "${MODEL}" \
  --base-url "http://127.0.0.1:${MODEL_PORT}/v1" \
  --api-key dummy \
  --judge-name smoke-judge \
  --judge-backend vllm \
  --judge-model "${JUDGE_MODEL}" \
  --judge-base-url "http://127.0.0.1:${JUDGE_PORT}/v1" \
  --judge-api-key dummy \
  --random-sample "${SAMPLE_SIZE}" \
  --sampling-seed 0

python3 -m eval_harness.cli run-all \
  --dataset-path eval_harness/data/EU_alert_working_copy/test/test.jsonl \
  --output-dir eval \
  --model-name smoke-model \
  --backend vllm \
  --model "${MODEL}" \
  --base-url "http://127.0.0.1:${MODEL_PORT}/v1" \
  --api-key dummy \
  --judge-name smoke-judge \
  --judge-backend vllm \
  --judge-model "${JUDGE_MODEL}" \
  --judge-base-url "http://127.0.0.1:${JUDGE_PORT}/v1" \
  --judge-api-key dummy \
  --judge-template-path eval_harness/prompts/judge_refusal.txt \
  --random-sample "${SAMPLE_SIZE}" \
  --sampling-seed 0 \
  --concurrency 4

kill "${JUDGE_PID}" "${MODEL_PID}" 2>/dev/null || true
echo "Smoke test finished. Artifacts under ${EU_GUARD_ROOT}/eval/"
