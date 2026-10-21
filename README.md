# Inference Engineering

Hands-on notes, benchmarks and code from a 4-week self-guided study of LLM inference engineering
(serving, KV cache management, batching, GPU utilization). Every entry includes raw data so results can be checked.

| Day | Topic | What's inside |
|---|---|---|
| [Day 1](day-01-vllm-baseline/) | vLLM baseline + PagedAttention | Qwen3-0.6B on 1x RTX 4090, concurrency sweep 1-256, raw `vllm bench serve` output |

Conventions: raw output is committed untouched; interpretation goes in each day's README and is labelled
*finding* (measured) or *hypothesis* (not yet verified).
