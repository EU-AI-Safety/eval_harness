#!/usr/bin/env bash

set -euo pipefail

MODEL_DIR="${VLLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-}"
PORT="${VLLM_PORT:-8000}"
HOST="${VLLM_HOST:-0.0.0.0}"
API_KEY="${VLLM_API_KEY:-dummy}"
DTYPE="${VLLM_DTYPE:-bfloat16}"
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"
TP_SIZE="${VLLM_TP_SIZE:-1}"
DP_SIZE="${VLLM_DP_SIZE:-1}"
ATTN_BACKEND="${VLLM_ATTN_BACKEND:-FLASHINFER}"
# Prefer explicit cache, then existing HF_HOME (HPC jobs), then home fallback.
CACHE_PATH="${VLLM_CACHE_PATH:-${HF_HOME:-${HOME}/.cache}}"
LOG_DIR="${VLLM_LOG_DIR:-eval/logs}"
LOG_FILE="${VLLM_LOG_FILE:-}"
DISABLE_PREFIX_CACHING="${VLLM_DISABLE_PREFIX_CACHING:-1}"
ENABLE_REQUEST_LOGGING="${VLLM_ENABLE_REQUEST_LOGGING:-1}"
EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
SHOW_HELP=0

usage() {
  cat <<'USAGE'
Usage: bash eval_harness/scripts/start_vllm_model_server.sh [model_dir_or_hf_repo]

Environment overrides:
  VLLM_MODEL                  Model path or HF repo id
  VLLM_SERVED_MODEL_NAME      Explicit served model name (default: same as VLLM_MODEL)
  VLLM_PORT                   Server port (default: 8000)
  VLLM_HOST                   Server host (default: 0.0.0.0)
  VLLM_API_KEY                API key required by the server (default: dummy)
  VLLM_DTYPE                  Model dtype (default: bfloat16)
  VLLM_GPU_MEM_UTIL           GPU memory utilization fraction (default: 0.85)
  VLLM_TP_SIZE                Tensor parallel size (default: 1)
  VLLM_DP_SIZE                Data parallel size (default: 1)
  VLLM_ATTN_BACKEND           FLASHINFER | FLASH_ATTN | TRITON_ATTN | FLEX_ATTENTION
  VLLM_CACHE_PATH             HF cache path (default: HF_HOME, else HOME/.cache)
  VLLM_LOG_DIR                Directory for server logs (default: eval/logs)
  VLLM_LOG_FILE               Explicit log file path
  VLLM_DISABLE_PREFIX_CACHING 1 to disable prefix caching (default: 1)
  VLLM_ENABLE_REQUEST_LOGGING 1 to enable request logging (default: 1)
  VLLM_EXTRA_ARGS             Extra raw args appended to the vLLM server command
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

# Default served name to the full model id so CLI --model matches /v1/models.
if [[ -z "$SERVED_MODEL_NAME" ]]; then
  SERVED_MODEL_NAME="$MODEL_DIR"
fi
LOG_NAME_SAFE="${SERVED_MODEL_NAME//\//_}"

case "$ATTN_BACKEND" in
  FLASHINFER|FLASH_ATTN|TRITON_ATTN|FLEX_ATTENTION|TORCH_SDPA) ;;
  *)
    echo "Error: unsupported VLLM_ATTN_BACKEND '$ATTN_BACKEND'" >&2
    exit 2
    ;;
esac

mkdir -p "$LOG_DIR"
if [[ -z "$LOG_FILE" ]]; then
  LOG_FILE="$LOG_DIR/vllm_model_${LOG_NAME_SAFE}_${DTYPE}_$(date +%Y%m%d_%H%M%S).log"
fi

COMMAND=(
  python3 -m vllm.entrypoints.openai.api_server
  --model "$MODEL_DIR"
  --host "$HOST"
  --port "$PORT"
  --api-key "$API_KEY"
  --served-model-name "$SERVED_MODEL_NAME"
  --dtype "$DTYPE"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --tensor-parallel-size "$TP_SIZE"
  --data-parallel-size "$DP_SIZE"
  --attention-backend "$ATTN_BACKEND"
  --override-generation-config '{"temperature": 0.0}'
)

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

echo "Starting vLLM model server"
echo "  model: $MODEL_DIR"
echo "  served_model_name: $SERVED_MODEL_NAME"
echo "  host: $HOST"
echo "  port: $PORT"
echo "  dtype: $DTYPE"
echo "  gpu_memory_utilization: $GPU_MEM_UTIL"
echo "  tensor_parallel_size: $TP_SIZE"
echo "  data_parallel_size: $DP_SIZE"
echo "  attention_backend: $ATTN_BACKEND"
echo "  override_generation_config: temperature=0.0"
echo "  prefix_caching_disabled: $DISABLE_PREFIX_CACHING"
echo "  request_logging_enabled: $ENABLE_REQUEST_LOGGING"
echo "  hf_cache_path: ${HF_HOME:-<unset>}"
echo "  log_file: $LOG_FILE"

exec "${COMMAND[@]}" >>"$LOG_FILE" 2>&1