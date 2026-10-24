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
- Weights: 1.40 GiB = 1.503 GB (checkpoint size from the log)
- KV read per sequence per step: average context 576 tokens (512 input + half of 128 output) x 114,688 B = 0.066 GB

Why memory-bound: one token costs about 2 x 0.6B = 1.2 GFLOPs, which takes roughly 10 microseconds on a GPU rated
well above 100 TFLOPS, versus ~1.5 ms to read 1.5 GB of weights at 1,008 GB/s (an order-of-magnitude estimate, not measured).

### Predicted vs measured

"Decode-only" = concurrency x 1000 / mean TPOT, which excludes prefill and the first-token wait and is therefore the
fairer comparison for a decode roofline. "Output" is the benchmark's end-to-end output token throughput.

| Concurrency | GB/step | Predicted tok/s | Measured output tok/s | Gap | Measured decode-only tok/s | Gap |
|---|---|---|---|---|---|---|
| 1  | 1.569 | 642   | 411   | 36% | 445   | 31% |
| 8  | 2.032 | 3,969 | 2,245 | 43% | 2,640 | 33% |
| 32 | 3.617 | 8,917 | 4,928 | 45% | 6,519 | 27% |

Gap = (predicted - measured) / predicted.

### Findings
- Batching behaves as the model predicts: the ideal rises ~14x from c=1 to c=32 because the weight read is shared, and measured throughput rises ~12x.
- Decode-only measurements sit at roughly 67-73% of the bandwidth ceiling at all three concurrency levels (gap 27-33%).
- The end-to-end gap grows with concurrency (36% to 45%).

### Hypotheses (not verified; no profiling done yet)
1. **Achievable bandwidth is below the quoted peak.** Real kernels typically reach a fraction of theoretical bandwidth.
2. **Per-kernel overheads.** A step runs many small kernels (28 layers, several ops each); at ~2 ms per step, fixed per-kernel costs are a visible share.
3. **Sampling and CPU/scheduler gaps** between steps are not counted in the formula.
4. **Prefill competes with decode** (explains the growing end-to-end gap): each request has 512 prompt tokens for only 128 output tokens, so a large share of GPU time goes to prompt processing that this decode formula ignores.
5. Attention cost beyond the simple KV-bytes model (kernel efficiency at small batch).

### Caveats
- The c=32 run lasted only 2.6 s and each point is a single run, so its decode-only number is approximate.
- Average context of 576 tokens is an approximation; the real average varies with how requests overlap.
- Weight bytes use the checkpoint size; the tied embedding / LM head matrix is read in full each step, so no adjustment was made.

## Next: profile to separate the hypotheses
Use Nsight Systems or `torch.profiler` on a c=1 decode loop to measure achieved bandwidth per kernel, kernel launch gaps,
and the CPU time between steps; then rerun with prefill excluded (e.g. very short prompts) to test hypothesis 4.
