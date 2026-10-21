#!/usr/bin/env bash
# Reproduces the Day 1 results. Server must already be running (see README).
# c=1,8,32 used 100 prompts; c=64,128,256 used 400 prompts so concurrency could exceed the prompt count.
set -euo pipefail
mkdir -p results
bench() {  # $1 = concurrency, $2 = num prompts
  vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
    --model Qwen/Qwen3-0.6B --dataset-name random \
    --random-input-len 512 --random-output-len 128 \
    --num-prompts "$2" --max-concurrency "$1" \
    --save-result --result-dir results --result-filename "c$1.json" \
  | tee "results/c$1.txt"
}
for C in 1 8 32;     do bench "$C" 100; done
for C in 64 128 256; do bench "$C" 400; done
