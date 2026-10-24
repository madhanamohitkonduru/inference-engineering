#!/usr/bin/env bash
# Quick sanity benchmark at c=1 and c=32 (same flags as Day 1, 100 prompts). Server must be running.
# Usage: bash scripts/bench_quick.sh [output_dir]
set -euo pipefail
if [ -f /workspace/env.sh ]; then source /workspace/env.sh; else source "$HOME/env.sh"; fi
OUT="${1:-$HOME/results-sanity}"; mkdir -p "$OUT"
for C in 1 32; do
  vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
    --model Qwen/Qwen3-0.6B --dataset-name random \
    --random-input-len 512 --random-output-len 128 \
    --num-prompts 100 --max-concurrency "$C" \
    --save-result --result-dir "$OUT" --result-filename "c${C}.json" > "$OUT/c${C}.txt" 2>&1
  echo "c=$C: $(grep 'Output token throughput (tok/s)' "$OUT/c${C}.txt")  $(grep 'Mean TPOT' "$OUT/c${C}.txt")"
done
echo "Day 1 reference (4090): c=1 410.8 tok/s, TPOT 2.25 ms; c=32 4927.5 tok/s, TPOT 4.91 ms"
