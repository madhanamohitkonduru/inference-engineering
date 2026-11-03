#!/usr/bin/env bash
# Day 10 concurrency sweep: random 512-in/128-out, c = 1..256, >= 500 prompts per point (10x c above c=50), per-request data saved.
# Server must be running (start it with: bash scripts/serve.sh --no-enable-prefix-caching).
# Usage: bash scripts/sweep_day10.sh <output_dir> <run_number>      e.g.  bash scripts/sweep_day10.sh ~/sweep run1
set -euo pipefail
if [ -f /workspace/env.sh ]; then source /workspace/env.sh; else source "$HOME/env.sh"; fi
OUT="${1:?output dir}"; RUN="${2:?run label, e.g. run1}"; mkdir -p "$OUT"
HELP="$( { vllm bench serve --help; vllm bench serve --help=all; } 2>&1 || true)"
for F in --save-detailed --percentile-metrics --metric-percentiles --goodput; do
  echo "$HELP" | grep -q -- "$F" || { echo "flag $F not found in 'vllm bench serve --help'; stop and check"; exit 1; }
done
COMMON=(--backend openai-chat --endpoint /v1/chat/completions --model Qwen/Qwen3-0.6B --dataset-name random
        --random-input-len 512 --random-output-len 128)
# warm-up (not saved)
vllm bench serve "${COMMON[@]}" --num-prompts 20 --max-concurrency 4 > /dev/null 2>&1
for C in 1 2 4 8 16 32 64 128 256; do
  N=$(( C * 10 > 500 ? C * 10 : 500 ))
  vllm bench serve "${COMMON[@]}" --num-prompts "$N" --max-concurrency "$C" \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 --goodput ttft:200 tpot:10 \
    --save-result --save-detailed --result-dir "$OUT" --result-filename "${RUN}_c${C}.json" > "$OUT/${RUN}_c${C}.txt" 2>&1
  echo "c=$C prompts=$N: $(grep 'Output token throughput (tok/s)' "$OUT/${RUN}_c${C}.txt")  $(grep -i 'P99 TTFT' "$OUT/${RUN}_c${C}.txt")  $(grep -i 'P99 TPOT' "$OUT/${RUN}_c${C}.txt")"
done
