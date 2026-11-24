# Day 14: Why one GPU isn't enough

Goal: answer with numbers when a model does not fit on one GPU, when it fits but leaves too little room for users, and what splitting it across GPUs buys and costs.
*Finding* = measured or read from a primary source; *hypothesis* = not verified. **Nothing on this page was measured on a multi-GPU machine**: it is arithmetic on published specs plus a reading of the vLLM code. Day 15 will measure it.
Sources for every spec number (config files, checkpoint sizes, NVIDIA pages) are in [`inputs.md`](inputs.md); raw config and index files are in `raw/`.

## What was built
- `memory_plan.py`: for a model, GPU, TP size, dtype, `gpu_memory_utilization` and context length, prints weights per GPU, KV bytes per token per GPU (replicate-heads rule: `2 x layers x max(1, kv_heads / TP) x head_dim x bytes`), KV capacity in tokens and sequences, and a verdict. Standard library only.
- `test_memory_plan.py`: first test **reproduces Day 3**: Qwen3-0.6B on the 4090 at TP=1 gives **167,790 tokens** (Day 3 corrected estimate 167,791; vLLM's own log 167,728, so within 0.1%), 114,688 KV bytes per token, and 1.192 GB of weights read per decode step, all matching Day 3. Other tests: the replicate-heads rule, an invalid TP size, verdicts, and the KV capacity more than doubling.
- `build_tables.py` (fit table, capacity table, doubling check) and `decode_model.py` (decode-time model and sensitivity); outputs are in `tables/`. `test_decode_model.py` checks the formula's pieces. All 11 tests pass: `python3 -m unittest test_memory_plan test_decode_model`.

## Self-check answers
1. **Why can TP make a decode step faster, and what pushes back?** All the GPUs each hold a slice of every layer and work at the same time. Decode is memory-bound (every step reads all the weights), so each GPU reading only 1/N of them makes the step shorter. What pushes back is that every layer ends with an all-reduce, a GPU-to-GPU meeting with a fixed delay (latency), and that adds time to every token.
2. **Why can going from 1 GPU to 2 give more than double the KV capacity?** The model's weights stay the same size, so splitting them across 2 GPUs leaves much more room for KV on each GPU: the weights are paid for once, not once per GPU. Each GPU also stores only half of each token's KV, so each token costs half as much. In the Qwen3-32B example below, 2x the GPUs gave 8.8x the capacity.
3. **Why is using more GPUs than KV heads wasteful?** A KV head cannot be split further, so each GPU keeps at least one. Beyond that, the heads are copied: each copy stores the same data (wasted memory) and some of the compute is repeated. vLLM does exactly this (see code notes).

## Predictions (written before calculating) and what the model says
| Question | My prediction | Result (arithmetic, not measured) |
|---|---|---|
| Q1. Qwen3-32B BF16 on one 24 GB 4090? On one 80 GB GPU? KV room left on 80 GB? | Does not fit on the 4090; fits on 80 GB with about 16 GB left at 100% utilization | Checkpoint is 65.5 GB (61.0 GiB). Does not fit on the 4090 (*finding*). Fits on an 80 GB GPU. At vLLM's default 0.9 utilization with 2 GiB assumed activation memory only 8.9 GiB is left for KV (4 sequences of 8,192); at 1.0 it would be 16.9 GiB, close to my 16 GB. |
| Q2. Fewest 80 GB GPUs for a 72B model; GPUs for 32 users at 8,192 tokens? | 144 GB so 2 GPUs; the 32-user part needs layers and head dimension | Checkpoint 145.4 GB, so 2 GPUs hold the weights, but TP=2 leaves room for only 1 sequence. TP=4 gives 57 sequences, so **4 GPUs** serve 32 users (TP=3 is not allowed: TP must divide the 64 attention heads). |
| Q3. 8B model on 2 GPUs: speedup with NVLink / with PCIe | NVLink 1.7x, PCIe 0.9x | Model, H100, batch 1: NVLink **1.51x**, PCIe **1.01x**. On a 4090 (PCIe only): **1.54x**. Day 15 measures the real numbers. |
| Q4. At TP=2, which matters more: extra bandwidth or all-reduce latency? | Extra bandwidth | The extra memory bandwidth is what creates the speedup; but among the interconnect assumptions, **latency decides how much of it survives**, and link bandwidth barely matters at batch 1 (see sensitivity). Day 15 measures it. |

## Fit and capacity tables (Step 5)
Assumptions: 8,192-token sequences, `gpu_memory_utilization` 0.9, **2 GiB activation memory per GPU (assumption, not measured)**, 0.06 GiB overhead (Day 3 value), weights split evenly across GPUs, quoted GPU memory treated as GiB. "tight" means it fits but holds fewer than 4 sequences (my own cutoff).

Fit table (8,192-token sequences, gpu_memory_utilization 0.9, activation 2.0 GiB assumed). no = doesn't fit; tight = fits but fewer than 4 sequences; FITS = at least 4 sequences. Cell = mark (sequences of 8,192).

| Model | GPU | TP=1 | TP=2 | TP=4 | TP=8 |
|---|---|---|---|---|---|
| qwen3-8b | rtx4090 | tight (3) | FITS (21) | FITS (55) | FITS (125) |
| qwen3-8b | a100-80 | FITS (48) | FITS (110) | FITS (235) | FITS (483) |
| qwen3-8b | h100-80 | FITS (48) | FITS (110) | FITS (235) | FITS (483) |
| qwen3-8b | h200 | FITS (97) | FITS (208) | FITS (430) | FITS (874) |
| qwen3-32b | rtx4090 | no | no | FITS (8) | FITS (47) |
| qwen3-32b | a100-80 | FITS (4) | FITS (39) | FITS (109) | FITS (249) |
| qwen3-32b | h100-80 | FITS (4) | FITS (39) | FITS (109) | FITS (249) |
| qwen3-32b | h200 | FITS (31) | FITS (94) | FITS (219) | FITS (468) |
| qwen2.5-72b | rtx4090 | no | no | no | FITS (8) |
| qwen2.5-72b | a100-80 | no | tight (1) | FITS (57) | FITS (169) |
| qwen2.5-72b | h100-80 | no | tight (1) | FITS (57) | FITS (169) |
| qwen2.5-72b | h200 | no | FITS (45) | FITS (145) | FITS (345) |

A100 80 GB and H100 80 GB rows are identical because the tables only count memory.

## Doubling check (Qwen3-32B on an 80 GB GPU, where TP=1 barely fits)
| TP | Weights per GPU (GiB) | KV heads per GPU | KV bytes/token/GPU | Free for KV (GiB) | KV tokens | Sequences of 8,192 | vs TP=1 |
|---|---|---|---|---|---|---|---|
| 1 | 61.02 | 8.0 | 262,144 | 8.92 | 36,519 | 4 | 1.0x |
| 2 | 30.51 | 4.0 | 131,072 | 39.43 | 322,993 | 39 | 8.84x |
| 4 | 15.26 | 2.0 | 65,536 | 54.68 | 895,941 | 109 | 24.53x |

*Finding* (arithmetic): capacity does more than double. Two effects multiply: more free memory on each GPU (8.9 to 39.4 GiB, 4.4x) and each GPU storing only part of every token's KV (half the bytes per token).

## Decode-time model (Qwen3-8B, 1,024 tokens of context, tokens per user per second)
`step time = (weight bytes read + KV bytes read) / memory bandwidth + (2 x layers + 2) x (latency + message bytes / link bandwidth)`; per GPU after splitting by TP. Weight bytes read are the checkpoint minus the embedding table (a decode step reads one row of it, not the whole table). The all-reduce latencies are **assumptions, not measurements: NVLink 10 us, PCIe 30 us**. I searched for a sourced small-message number and found none, so they are guesses (a direct GPU-to-GPU link versus PCIe's extra hops); Day 15 measures them. Link bandwidth per direction: PCIe 4.0 x16 31.5 GB/s (arithmetic from the PCIe 4.0 rate; NVIDIA's "64 GB/s" is both directions), NVLink 450 GB/s (*hypothesis*: half of NVIDIA's quoted 900 GB/s, supported only by a Runpod guide calling it bidirectional).

Batch 1:

| GPU and link | TP=1 | TP=2 | TP=4 | TP=8 |
|---|---|---|---|---|
| H100 SXM, NVLink (assumed 10 us) | 219.1 | 330.8 (1.51x) | 531.3 (2.42x) | 762.3 (3.48x) |
| H100 SXM, PCIe (assumed 30 us) | 219.1 | 221.2 (1.01x) | 295.8 (1.35x) | 355.9 (1.62x) |
| RTX 4090, PCIe (assumed 30 us) | 65.9 | 101.8 (1.54x) | 165.8 (2.51x) | 241.8 (3.67x) |
| RTX 4090, NVLink (hypothetical: the 4090 has none) | 65.9 | 120.1 (1.82x) | 220.6 (3.35x) | 379.2 (5.75x) |

Batch 32:

| GPU and link | TP=1 | TP=2 | TP=4 | TP=8 |
|---|---|---|---|---|
| H100 SXM, NVLink (assumed 10 us) | 167.8 | 265.7 (1.58x) | 439.9 (2.62x) | 654.4 (3.9x) |
| H100 SXM, PCIe (assumed 30 us) | 167.8 | 171.9 (1.02x) | 231.2 (1.38x) | 279.3 (1.66x) |
| RTX 4090, PCIe (assumed 30 us) | 50.5 | 78.5 (1.55x) | 128.4 (2.54x) | 188.2 (3.73x) |
| RTX 4090, NVLink (hypothetical: the 4090 has none) | 50.5 | 93.6 (1.85x) | 174.3 (3.45x) | 306.8 (6.08x) |

The 4090 NVLink row is hypothetical. TP=8 rows are too optimistic because the same latency is used for every TP size (real latency grows with more GPUs).

*Finding* (from the model): on a fast GPU like the H100, PCIe gives almost no speedup at TP=2 (1.01x) because the 74 all-reduces per step cost about as much as the weight reading they save; on a slow-memory 4090 the same PCIe cost is small next to a 15 ms step, so TP=2 still gives 1.54x.

## Sensitivity (H100, NVLink assumptions, TP=2)
| Changed assumption (H100, NVLink, TP=2) | Batch 1: tok/s (fraction of baseline) | Batch 32 |
|---|---|---|
| all-reduce latency 2 us | 411.3 (1.244) | 315.3 (1.187) |
| all-reduce latency 5 us | 376.9 (1.139) | 294.7 (1.109) |
| all-reduce latency 10 us | 330.8 (1.0) | 265.7 (1.0) |
| all-reduce latency 20 us | 265.7 (0.803) | 222.0 (0.836) |
| all-reduce latency 40 us | 190.7 (0.577) | 167.1 (0.629) |
| all-reduce latency 80 us | 121.9 (0.369) | 111.8 (0.421) |
| link bandwidth 30 GB/s | 328.7 (0.994) | 229.0 (0.862) |
| link bandwidth 100 GB/s | 330.3 (0.998) | 255.5 (0.961) |
| link bandwidth 450 GB/s | 330.8 (1.0) | 265.7 (1.0) |
| link bandwidth 1800 GB/s | 330.9 (1.0) | 268.0 (1.009) |

*Finding* (from the model): latency matters a lot, link bandwidth barely matters at batch 1 (a 30 GB/s link costs under 1%), but at batch 32 the messages are larger and a slow link costs about 14%.

## Code notes (Step 6, vLLM v0.32.0)
See [`code_notes.md`](code_notes.md). Short version: `qkv_proj` and `gate_up_proj` are column-parallel (no communication); `o_proj` and `down_proj` are `RowParallelLinear`, which calls `tensor_model_parallel_all_reduce`, so two all-reduces per layer. There are two more communications per step (an all-reduce after the embedding lookup and a gather of the logits), so the count is 2 x layers + 2, which I changed in `decode_model.py`. The replicate-heads rule in `memory_plan.py` matches the code (`num_kv_heads = max(1, total_kv_heads // tp)`), so it needed no change.

## 2-GPU pods on Runpod (Step 7)
See [`pod_options.md`](pod_options.md). RTX 4090 pods: $0.89 per GPU-hour Secure, so about $1.78/hr for two; no NVLink, so PCIe only. H100 SXM: $4.39 per GPU-hour Secure, about $8.78/hr for two; NVLink 4.0 per Runpod's guide. Suggested Day 15 plan: 2x 4090 for the PCIe case (about 1.5 hours, roughly $2.70) and a short 2x H100 SXM session for the NVLink case (about $9 per hour). Prices are from Runpod's pricing page dated 2026-11-12; availability was not checked.

## What I did not do
- No measurement on any multi-GPU machine. Every speed number is a model with assumed latencies.
- The two all-reduce latencies (10 us, 30 us) and the 2 GiB activation memory are guesses; the 4-sequence "too little KV" cutoff is my own.
- NVIDIA's pages do not say whether NVLink and PCIe figures are per direction or both directions; the 450 GB/s per-direction NVLink figure is a hypothesis, and the PCIe 4.0 rate comes from news coverage of the PCI-SIG spec, not the spec itself.
- The A100 80 GB bandwidth (2,039 GB/s) came from a garbled datasheet table; I did not confirm whether it is the SXM or PCIe variant.
- I did not model compute time, the ring all-reduce factor, the larger message of the logits gather, or any fusion of all-reduces with the next layer's normalization.
- I read the Qwen3 code only; the 72B model uses the Qwen2 model file, which I did not read. I did not read how the vocabulary shard is split.
- Pipeline parallelism, FP8 (the script supports it but no tables use it) and expert parallelism are not covered.
- I did not check live Runpod availability for 2-GPU pods or whether two SXM GPUs in one pod are NVLinked (`nvidia-smi topo -m` on the pod will show it).
