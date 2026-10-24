# Day 3: KV-cache math and a decode roofline

Model: `Qwen/Qwen3-0.6B` (BF16), GPU: RTX 4090 (24 GB), vLLM 0.32.0. Measured data comes from the
[Day 1 benchmark](../day-01-vllm-baseline/). Everything below is reproduced by [`roofline.py`](roofline.py)
(`python3 roofline.py`, standard library only).

## 1. KV cache math

From `config.json`: `num_hidden_layers = 28`, `num_key_value_heads = 8`, `head_dim = 128`
(read from the `head_dim` field; `hidden_size / num_attention_heads` would give a different number for this model).

```
KV bytes per token = 2 (K and V) x layers x kv_heads x head_dim x bytes_per_element
                   = 2 x 28 x 8 x 128 x 2 = 114,688 bytes (112 KiB)
KV bytes for a 1,000-token sequence = 114,688,000 bytes (~109 MiB)
```

**How many tokens fit?** Inputs from the vLLM startup log: GPU memory 23.52 GiB, `gpu_memory_utilization` 0.85, weights 1.40 GiB.

| Method | KV memory | Tokens | vs vLLM log (167,728) |
|---|---|---|---|
| Naive: 0.85 x 23.52 - weights | 18.59 GiB | 174,064 | +3.8% |
| Corrected: budget - consumed (1.46) - peak activation (0.61) | 17.92 GiB | 167,791 | +0.04% |
| vLLM log | 17.92 GiB | 167,728 | n/a |

The naive estimate is 3.8% high because it ignores memory vLLM sets aside before sizing the cache: peak activation
memory (scratch space for a forward pass, profiled at startup with a dummy full-size batch) and ~0.06 GiB of non-weight
overhead. With those subtracted, the estimate matches the log (the small remainder is rounding of the logged GiB values).

Capacity in sequences: **167** sequences of 1,000 tokens (167,728 / 1,000). A system that reserved the full
`max_model_len` of 4,096 tokens per request would fit **40** (the log's "40.95x"), about 4x fewer, which is the
reservation/internal-fragmentation waste that paged allocation avoids.

## 2. Decode roofline

Decode produces one token per sequence per step. Each step must read all weights once, plus each sequence's KV cache.
If decode is memory-bound, the ceiling is:

```
tokens/s = (sequences in batch) / ((weight_bytes + sum of KV bytes read) / memory_bandwidth)
```

Inputs:
- Bandwidth: RTX 4090 = 21 Gbps x 384-bit / 8 = **1,008 GB/s** (NVIDIA spec)
- Weight bytes read per step: **1.192 GB** = layer matmul weights 880.8 MB (28 layers x 4 matrices) + `lm_head` 311.2 MB
  (151,936 x 1,024 x 2 B). Shapes from `config.json`: hidden 1024, intermediate 3072, 16 heads, 8 KV heads, head_dim 128, vocab 151,936.
- KV read per sequence per step: average context 576 tokens (512 input + half of 128 output) x 114,688 B = 0.066 GB

**Correction (Oct 24).** The first version of this write-up used the 1.503 GB checkpoint size as weight bytes. That overcounts:
the embedding table is only indexed (one row per token), so a decode step reads about 1.19 GB, not 1.50 GB. The Day 4 profile
supports this: it shows 112 layer matmuls plus one lm_head matmul per step, with the lm_head kernel moving ~311 MB at ~950 GB/s.
The corrected ceiling is higher and the gap larger; the superseded figures are listed below for transparency.

Why memory-bound: one token costs about 2 x 0.6B = 1.2 GFLOPs, which takes roughly 10 microseconds on a GPU rated
well above 100 TFLOPS, versus ~1.2 ms to read 1.2 GB of weights at 1,008 GB/s (an order-of-magnitude estimate).

### Predicted vs measured

"Decode-only" = concurrency x 1000 / mean TPOT (excludes prefill and the first-token wait), the fairer comparison for a
decode roofline. "Output" is the benchmark's end-to-end output token throughput. Gap = (predicted - measured) / predicted.

| Concurrency | GB/step | Predicted tok/s | Measured output tok/s | Gap | Measured decode-only tok/s | Gap |
|---|---|---|---|---|---|---|
| 1  | 1.258 | 801   | 411   | 49% | 445   | 44% |
| 8  | 1.720 | 4,687 | 2,245 | 52% | 2,640 | 44% |
| 32 | 3.306 | 9,757 | 4,928 | 49% | 6,519 | 33% |

Superseded first estimate (checkpoint size as weight bytes): predicted 642 / 3,969 / 8,917 tok/s, decode-only gaps 31% / 33% / 27%.

### Findings
- Batching behaves as the model predicts: the ideal rises ~12x from c=1 to c=32 (weights are read once per step for the whole batch) and measured throughput rises ~12x.
- Decode-only measurements reach roughly 56% (c=1, c=8) to 67% (c=32) of the bandwidth ceiling.

### Where the gap comes from
Measured on Day 4 with the PyTorch profiler and Nsight Systems; see [`../day-04-profiling/`](../day-04-profiling/).
At c=1 the GPU is busy ~95% of the time, so the gap is mostly kernel time, not idle: layer matmuls reach ~75% of peak bandwidth,
attention only ~17% (many tiny kernels), and ~0.4 ms per step goes to small kernels (norms, sampling, cache writes).

### Caveats
- The c=32 run lasted only 2.6 s and each point is a single run, so its decode-only number is approximate.
- Average context of 576 tokens is an approximation; the real average varies with how requests overlap.
- Prefill competes with decode in the end-to-end numbers; this was not measured separately here.
