#!/usr/bin/env bash
# Start vLLM with the same settings as Days 1-3. Extra args are passed through, e.g.:  bash scripts/serve.sh --enforce-eager
set -euo pipefail
if [ -f /workspace/env.sh ]; then source /workspace/env.sh; else source "$HOME/env.sh"; fi
LOG_DIR="${LOG_DIR:-$HOME/logs}"; mkdir -p "$LOG_DIR"
vllm serve Qwen/Qwen3-0.6B --max-model-len 4096 --gpu-memory-utilization 0.85 --port 8000 "$@" \
  2>&1 | tee "$LOG_DIR/server_$(date +%H%M%S).log"
