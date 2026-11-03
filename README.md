# Inference Engineering

Hands-on notes, benchmarks and code from a 4-week self-guided study of LLM inference engineering
(serving, KV cache management, batching, GPU utilization). Every entry includes raw data so results can be checked.

| Day | Topic | What's inside |
|---|---|---|
| [Day 1](day-01-vllm-baseline/) | vLLM baseline + PagedAttention | Qwen3-0.6B on 1x RTX 4090, concurrency sweep 1-256, raw `vllm bench serve` output |
| [Day 3](day-03-roofline/) | KV-cache math + decode roofline | Hand-derived KV bytes/token and capacity vs vLLM's log; predicted vs measured tokens/s at c=1/8/32 |
| [Day 4](day-04-profiling/) | Profiling decode | PyTorch profiler + Nsight Systems on Qwen3-0.6B: where a decode step goes, CUDA graphs on vs off |
| [Day 5](day-05-continuous-batching/) | Continuous batching | `--max-num-seqs` 1 vs 4 with 1 long + 3 short requests; walk through vLLM's `schedule()` |
| [Day 6](day-06-prefix-caching/) | Prefix caching | Hit rates in the Day 1 benchmark, TTFT with caching on/off, edge cases (changed token, shared text last, short prefix), cache code walk-through |
| [Day 7](day-07-chunked-prefill/) | Chunked prefill | 6 decoding users + one 3,500-token prompt at `max_num_batched_tokens` 8192 / 2048 / 512: stall vs the long prompt's TTFT |
| [Day 8](day-08-quantization/) | Quantization | BF16 vs FP8 weights vs FP8 weights + FP8 KV cache: memory, TPOT at c=1 and c=32, and output changes on 50 prompts (4-bit not measured) |
| [Day 9](day-09-speculative-decoding/) | Speculative decoding (n-gram) | Copy-heavy vs free-writing prompts at c=1 and c=32, k sweep 2 to 16, acceptance rate, async-scheduling confound, and a 50-prompt output-identity check |
| [Day 10](day-10-goodput-analysis/) | Percentiles, goodput and capacity | pandas and DuckDB analysis of the Day 9 data, then a c=1 to 256 sweep with per-request TTFT/TPOT: latency-vs-throughput curve, goodput under an SLO, capacity (16 users per 4090 at p99 TTFT <= 200 ms), prefix caching on vs off |

## Rebuilding the environment on a fresh GPU pod
```bash
cd /workspace && git clone https://github.com/madhanamohitkonduru/inference-engineering
bash inference-engineering/scripts/setup.sh      # vLLM 0.32.0, profiling tools, helper env file
source /workspace/env.sh                         # in every new terminal
bash inference-engineering/scripts/serve.sh      # same server settings as Days 1-3
bash inference-engineering/scripts/bench_quick.sh  # c=1 and c=32 sanity check
```

Conventions: raw output is committed untouched; interpretation goes in each day's README and is labelled
*finding* (measured) or *hypothesis* (not yet verified).
