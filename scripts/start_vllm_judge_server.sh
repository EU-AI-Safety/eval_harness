#!/usr/bin/env bash

set -euo pipefail

MODEL_DIR="${VLLM_JUDGE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SERVED_MODEL_NAME="${VLLM_JUDGE_SERVED_MODEL_NAME:-}"
PORT="${VLLM_JUDGE_PORT:-8001}"
HOST="${VLLM_JUDGE_HOST:-0.0.0.0}"
API_KEY="${VLLM_JUDGE_API_KEY:-dummy}"
DTYPE="${VLLM_JUDGE_DTYPE:-bfloat16}"
GPU_MEM_UTIL="${VLLM_JUDGE_GPU_MEM_UTIL:-0.85}"
TP_SIZE="${VLLM_JUDGE_TP_SIZE:-1}"
DP_SIZE="${VLLM_JUDGE_DP_SIZE:-1}"
# Empty = let vLLM choose (needed for Gemma-4 heterogeneous head dims).
ATTN_BACKEND="${VLLM_JUDGE_ATTN_BACKEND:-}"
# Prefer explicit cache, then existing HF_HOME (HPC jobs), then home fallback.
CACHE_PATH="${VLLM_JUDGE_CACHE_PATH:-${HF_HOME:-${HOME}/.cache}}"
LOG_DIR="${VLLM_JUDGE_LOG_DIR:-eval/logs}"
LOG_FILE="${VLLM_JUDGE_LOG_FILE:-}"
DISABLE_PREFIX_CACHING="${VLLM_JUDGE_DISABLE_PREFIX_CACHING:-1}"
ENABLE_REQUEST_LOGGING="${VLLM_JUDGE_ENABLE_REQUEST_LOGGING:-1}"
EXTRA_ARGS="${VLLM_JUDGE_EXTRA_ARGS:-}"
SHOW_HELP=0

usage() {
  cat <<'USAGE'
Usage: bash eval_harness/scripts/start_vllm_judge_server.sh [model_dir_or_hf_repo]

Environment overrides:
  VLLM_JUDGE_MODEL                   Model path or HF repo id
  VLLM_JUDGE_SERVED_MODEL_NAME       Explicit served model name (default: same as VLLM_JUDGE_MODEL)
  VLLM_JUDGE_PORT                    Server port (default: 8001)
  VLLM_JUDGE_HOST                    Server host (default: 0.0.0.0)
  VLLM_JUDGE_API_KEY                 API key required by the server (default: dummy)
  VLLM_JUDGE_DTYPE                   Model dtype (default: bfloat16)
  VLLM_JUDGE_GPU_MEM_UTIL            GPU memory utilization fraction (default: 0.85)
  VLLM_JUDGE_TP_SIZE                 Tensor parallel size (default: 1)
  VLLM_JUDGE_DP_SIZE                 Data parallel size (default: 1)
  VLLM_JUDGE_ATTN_BACKEND            FLASHINFER | FLASH_ATTN | TRITON_ATTN | FLEX_ATTENTION
  VLLM_JUDGE_CACHE_PATH              HF cache path (default: HF_HOME, else HOME/.cache)
  VLLM_JUDGE_LOG_DIR                 Directory for server logs (default: eval/logs)
  VLLM_JUDGE_LOG_FILE                Explicit log file path
  VLLM_JUDGE_DISABLE_PREFIX_CACHING  1 to disable prefix caching (default: 1)
  VLLM_JUDGE_ENABLE_REQUEST_LOGGING  1 to enable request logging (default: 1)
  VLLM_JUDGE_EXTRA_ARGS              Extra raw args appended to the vLLM server command
USAGE
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  SHOW_HELP=1
elif [[ $# -gt 0 ]]; then
  MODEL_DIR="$1"
fi

if [[ $SHOW_HELP -eq 1 ]]; then
  usage
  exit 0
fi

# Default served name to the full model id so CLI --judge-model matches /v1/models.
if [[ -z "$SERVED_MODEL_NAME" ]]; then
  SERVED_MODEL_NAME="$MODEL_DIR"
fi
LOG_NAME_SAFE="${SERVED_MODEL_NAME//\//_}"

case "$ATTN_BACKEND" in
  ""|FLASHINFER|FLASH_ATTN|TRITON_ATTN|FLEX_ATTENTION|TORCH_SDPA) ;;
  *)
    echo "Error: unsupported VLLM_JUDGE_ATTN_BACKEND '$ATTN_BACKEND'" >&2
    exit 2
    ;;
esac

mkdir -p "$LOG_DIR"
if [[ -z "$LOG_FILE" ]]; then
  LOG_FILE="$LOG_DIR/vllm_judge_${LOG_NAME_SAFE}_${DTYPE}_$(date +%Y%m%d_%H%M%S).log"
fi

# Launch via Gemma-4 hetero compat shim (no-op for homogeneous models).
# transformers>=5.14 remaps dual head dims into per_layer_config; vLLM 0.26
# still needs legacy global_head_dim access for google/gemma-4-* judges.
COMMAND=(
  python3 -m eval_harness.vllm_gemma4_hetero_compat --
  -m vllm.entrypoints.openai.api_server
  --model "$MODEL_DIR"
  --host "$HOST"
  --port "$PORT"
  --api-key "$API_KEY"
  --served-model-name "$SERVED_MODEL_NAME"
  --dtype "$DTYPE"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --tensor-parallel-size "$TP_SIZE"
  --data-parallel-size "$DP_SIZE"
  --override-generation-config '{"temperature": 0.0}'
)

if [[ -n "$ATTN_BACKEND" ]]; then
  COMMAND+=(--attention-backend "$ATTN_BACKEND")
fi

if [[ "$DISABLE_PREFIX_CACHING" == "1" ]]; then
  COMMAND+=(--no-enable-prefix-caching)
fi

if [[ "$ENABLE_REQUEST_LOGGING" == "1" ]]; then
  COMMAND+=(--enable-log-requests)
fi

if [[ -n "$CACHE_PATH" ]]; then
  export HF_HOME="$CACHE_PATH"
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARRAY=($EXTRA_ARGS)
  COMMAND+=("${EXTRA_ARGS_ARRAY[@]}")
fi

echo "Starting vLLM judge server"
echo "  model: $MODEL_DIR"
echo "  served_model_name: $SERVED_MODEL_NAME"
echo "  host: $HOST"
echo "  port: $PORT"
echo "  dtype: $DTYPE"
echo "  gpu_memory_utilization: $GPU_MEM_UTIL"
echo "  tensor_parallel_size: $TP_SIZE"
echo "  data_parallel_size: $DP_SIZE"
echo "  attention_backend: ${ATTN_BACKEND:-<vllm default>}"
echo "  override_generation_config: temperature=0.0"
echo "  prefix_caching_disabled: $DISABLE_PREFIX_CACHING"
echo "  request_logging_enabled: $ENABLE_REQUEST_LOGGING"
echo "  hf_cache_path: ${HF_HOME:-<unset>}"
echo "  log_file: $LOG_FILE"

exec "${COMMAND[@]}" >>"$LOG_FILE" 2>&1