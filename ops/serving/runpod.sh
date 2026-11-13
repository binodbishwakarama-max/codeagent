#!/usr/bin/env bash
# ==============================================================================
# RunPod / Cloud GPU Launch Script for Prism Review vLLM Serving
#
# Supported hardware:
#   - NVIDIA A10G (24GB VRAM) -> recommended with --quantization awq
#   - NVIDIA L4 (24GB VRAM)   -> recommended with --quantization awq
#   - NVIDIA A100 (40GB/80GB) -> recommended for 16-bit BF16 (no quantization)
#
# Usage:
#   ./ops/serving/runpod.sh
#   ./ops/serving/runpod.sh --model Qwen/Qwen2.5-Coder-7B-Instruct --quantization awq --port 8000
# ==============================================================================

set -euo pipefail

# Default configuration values
MODEL_PATH="${MODEL_PATH:-models/qwen2.5-coder-7b-prism/merged_16bit}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
QUANTIZATION="${QUANTIZATION:-}"

# Parse optional command-line flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --gpu-memory-utilization)
      GPU_MEM_UTIL="$2"
      shift 2
      ;;
    --quantization)
      QUANTIZATION="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--model PATH] [--port PORT] [--quantization awq|gptq] [--max-model-len LEN]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "===================================================================="
echo "  Starting Prism Review vLLM Inference Server"
echo "===================================================================="
echo "Model Path:               ${MODEL_PATH}"
echo "Binding Address:          http://${HOST}:${PORT}"
echo "Max Sequence Length:      ${MAX_MODEL_LEN}"
echo "GPU Memory Utilization:   ${GPU_MEM_UTIL}"
echo "Quantization:             ${QUANTIZATION:-None (16-bit)}"
echo "===================================================================="

# 1. Verify NVIDIA GPU and driver availability
if ! command -v nvidia-smi &> /dev/null; then
  echo "[ERROR] nvidia-smi not found. Ensure an NVIDIA GPU is attached and drivers are installed."
  exit 1
fi

echo "[INFO] Detected NVIDIA GPU(s):"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# 2. Verify vLLM installation
if ! command -v vllm &> /dev/null; then
  echo "[WARNING] vllm executable not found in PATH."
  echo "[INFO] Attempting to install vLLM via pip..."
  pip install --no-cache-dir vllm
fi

# 3. Construct launch arguments
VLLM_ARGS=(
  serve
  "${MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --trust-remote-code
)

if [[ -n "${QUANTIZATION}" ]]; then
  VLLM_ARGS+=(--quantization "${QUANTIZATION}")
fi

echo "[INFO] Executing: vllm ${VLLM_ARGS[*]}"
exec vllm "${VLLM_ARGS[@]}"

