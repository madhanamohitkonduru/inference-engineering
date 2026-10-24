# Inference Engineering

Hands-on notes, benchmarks and code from a 4-week self-guided study of LLM inference engineering
(serving, KV cache management, batching, GPU utilization). Every entry includes raw data so results can be checked.

| Day | Topic | What's inside |
|---|---|---|
| [Day 1](day-01-vllm-baseline/) | vLLM baseline + PagedAttention | Qwen3-0.6B on 1x RTX 4090, concurrency sweep 1-256, raw `vllm bench serve` output |
| [Day 3](day-03-roofline/) | KV-cache math + decode roofline | Hand-derived KV bytes/token and capacity vs vLLM's log; predicted vs measured tokens/s at c=1/8/32 |

Conventions: raw output is committed untouched; interpretation goes in each day's README and is labelled
*finding* (measured) or *hypothesis* (not yet verified).
