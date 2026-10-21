# Day 1: vLLM baseline and PagedAttention

## Setup
- **Hardware:** 1x NVIDIA RTX 4090 (24 GB), rented on RunPod (PyTorch 2.8 / CUDA 12.8 template)
- **Software:** vLLM 0.32.0, torch 2.13.0, transformers 5.19.0 (see `results/env.txt`)
- **Model:** `Qwen/Qwen3-0.6B`
- **Server:**
  ```bash
  vllm serve Qwen/Qwen3-0.6B --max-model-len 4096 --gpu-memory-utilization 0.85 --port 8000
  ```
- **Workload:** `vllm bench serve`, random dataset, 512 input / 128 output tokens, OpenAI chat endpoint.
  c=1, 8, 32 used 100 prompts; c=64, 128, 256 used 400 prompts. Script: [`run_benchmark.sh`](run_benchmark.sh).

## Server startup log (ties to the PagedAttention paper)
- Available KV cache memory: **17.92 GiB**
- GPU KV cache size: **167,728 tokens**
- Maximum concurrency for 4,096 tokens/request: **40.95x** (= 167,728 / 4,096; the worst case where every
  request uses its full context)
- Cross-check: Qwen3-0.6B stores ~112 KiB of KV per token (2 x 28 layers x 8 KV heads x 128 dim x 2 bytes);
  167,728 tokens x 112 KiB ~ 17.9 GiB.

## Results
| Concurrency | Output tok/s | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | Run length (s) |
|---|---|---|---|---|---|
| 1   | 410.8  | 25.3   | 31.8   | 2.25  | 31.2 |
| 8   | 2244.6 | 55.4   | 71.2   | 3.03  | 5.7  |
| 32  | 4927.5 | 105.7  | 171.6  | 4.91  | 2.6  |
| 64  | 6477.3 | 148.6  | 405.9  | 8.32  | 7.9  |
| 128 | 6848.7 | 295.4  | 1013.0 | 15.35 | 7.5  |
| 256 | 6698.1 | 1069.1 | 2603.5 | 25.72 | 7.6  |

All 1,500 requests succeeded. Raw output: [`results/`](results/) (`cN.txt` console summary, `cN.json` saved result).

## Findings
- Throughput rises ~16x from c=1 to c=64, then plateaus at ~6.5-6.8k output tok/s (c=64/128/256 differ by
  less than run-to-run noise).
- Past the plateau, extra concurrency only adds queueing: from c=64 to c=256, mean TTFT is ~7x and TPOT ~3x
  worse at flat throughput. By Little's law, per-request time goes from ~1.3 s (64 / 50.6 req/s) to ~4.9 s (256 / 52.3 req/s).
- The knee is around c=64: nearly all the throughput for a modest latency cost.

## Hypotheses (not verified)
- Batching helps because each decode step reads the model weights once regardless of batch size.
- Up to c=32, step time grows mainly with per-request KV reads. From c=32 to c=64 TPOT rose ~1.7x for a 2x batch,
  which looks closer to compute saturation, with prefill interleaved into decode steps, and possibly CPU-side scheduling overhead on such a small model.
- KV capacity: at c=64 the working set (~64 x 640 = ~41k tokens) is far below the 167,728-token pool. At c=256 it
  (~164k tokens) approaches capacity, so the top of the sweep may also be affected by KV pressure or by the scheduler's
  `max_num_seqs` limit. Not measured.

## Caveats
- Runs last only 2.6-31 s; means are noisy and each point is a single run.
- Synthetic random prompts; sampling used the server defaults (the bench tool no longer forces temperature 0), and Qwen3 emits `<think>` tokens.
- The "Peak concurrent requests" line in the tool output exceeds the configured concurrency, so it is measured
  differently from its name and is not used here.
- Single GPU, single model, small model.

## PagedAttention notes (Kwon et al., SOSP 2023, arXiv 2309.06180)
KV cache grows linearly with generated tokens and the final length is unknown. Older systems reserved a contiguous
max-length chunk per request, wasting memory three ways: **reservation** (slots held but not yet used),
**internal fragmentation** (slots never used), **external fragmentation** (gaps between chunks too awkward to reuse).
PagedAttention stores the cache in small fixed-size blocks allocated on demand, mapped by a block table (like OS paging),
so waste is limited to the last partially filled block of each request.

## Next
- Vary `--max-num-seqs` and `--max-num-batched-tokens`; use longer runs and repeat each point.
- Profile GPU utilization during the plateau to test the compute-bound hypothesis.
