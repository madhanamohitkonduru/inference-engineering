# Day 4: profiling decode on an RTX 4090 (PyTorch profiler + Nsight Systems)

Goal: explain the gap between the Day 3 memory-bandwidth roofline and the measured decode speed of Qwen3-0.6B.

## Setup
- Fresh RunPod pod (no persistent volume), **RTX 4090, driver 595.91.07**, vLLM 0.32.0, Nsight Systems 2024.6.2.
  Environment rebuilt with one command: [`scripts/setup.sh`](../scripts/setup.sh) (+ `serve.sh`, `bench_quick.sh`).
- Same server settings as Days 1-3: `--max-model-len 4096 --gpu-memory-utilization 0.85`.
- **Sanity check vs Day 1** (100 prompts): c=1 428.7 vs 410.8 tok/s (TPOT 2.26 vs 2.25 ms); c=32 5,636 vs 4,928 tok/s
  (TPOT 4.71 vs 4.91 ms). TPOT agrees within 4%; the c=32 throughput difference reflects a ~2.5 s run being noisy. KV pool unchanged (167,728 tokens).

## 1. PyTorch profiler (graphs on)
Commands and full summaries: [`pytorch-profiler/`](pytorch-profiler/). Recorded 300 iterations, no Python stacks.

| | c=1 | c=32 |
|---|---|---|
| GPU busy (union of kernel intervals) | 95.3% | 95.4% |
| Attention share of kernel time | 18.8% | 49.5% |
| GEMM share | 65.3% | 38.7% |
| Other small kernels | 15.9% | 11.8% |
| Top kernel | `gemvx` (matrix-vector), 64.7% | `flash_fwd_splitkv`, 44.1% |

- The GPU is almost never idle with CUDA graphs on: the gap to the roofline is kernel time, not launch gaps.
- c=1 uses matrix-vector (`gemvx`) kernels; c=32 uses tensor-core matrix-matrix (`cutlass`/`ampere gemm`) kernels.
- At c=32, attention takes ~2.8 ms/step and reads ~2.1 GB of KV per step, about 75% of peak bandwidth (approximate: step count inferred from call counts), versus ~17% at c=1.
- Matrix work costs about the same per step at c=32 (~1.4 ms) as at c=1 (~1.5 ms): the weight read is shared by 32 sequences.

## 2. Where one decode step goes at c=1
Produced by [`decompose.py`](decompose.py) from the profiler totals. Bytes per part come from `config.json` shapes (verified):
layers 880.8 MB, lm_head 311.2 MB, KV 66.1 MB (576-token average context).

| Part | Time/step (ms) | Ideal at 1,008 GB/s (ms) | Extra (ms) | Share of extra | Achieved GB/s |
|---|---|---|---|---|---|
| Layer matmuls (112/step) | 1.157 | 0.874 | +0.283 | 26% | 761 (76%) |
| lm_head matmul | 0.328 | 0.309 | +0.019 | 2% | 949 (94%) |
| Attention (56 kernels/step) | 0.391 | 0.066 | +0.326 | 29% | 169 (17%) |
| Other small kernels | 0.377 | 0.000 | +0.377 | 34% | n/a |
| Idle gaps (Nsight) | 0.099 | 0.000 | +0.099 | 9% | n/a |
| **Total** | **2.352** | **1.248** | **+1.104** | 100% | |

The profile-based total (2.35 ms) is within 4% of the unprofiled TPOT (2.26 ms), so this accounts for essentially all of the
Day 3 c=1 gap (ideal 1.26 ms/step vs measured 2.26 ms; decode-only gap 44%).

**Finding:** the larger a kernel's data volume, the closer it runs to peak bandwidth (lm_head 94%, layer matmuls 76%, attention 17%).
**Hypothesis (not measured):** the small kernels are limited by fixed per-kernel start-up/tail cost and by too few thread blocks to fill the 128 SMs.

## 3. CUDA graphs on vs off (Nsight Systems)
Full output: [`nsight/summaries.md`](nsight/summaries.md).

| Per decode step | c=1 graphs ON | c=1 graphs OFF | c=32 graphs ON | c=32 graphs OFF |
|---|---|---|---|---|
| Time per step | 2.37 ms | 10.06 ms | 5.34 ms | 11.53 ms |
| GPU busy | 2.28 ms | 2.31 ms | 5.10 ms | 5.29 ms |
| GPU idle | 0.10 ms | 7.75 ms | 0.24 ms | 6.24 ms |
| Idle share | 4% | 77% | 4.5% | 54% |

Plain benchmark without the profiler (`--enforce-eager`): c=1 TPOT 2.26 -> 7.59 ms (**3.4x slower**), c=32 TPOT 4.71 -> 8.59 ms (**1.8x slower**).

- The GPU's own work is identical with and without graphs; all the extra time is the GPU waiting for the CPU to launch ~390-490 kernels per step one by one.
- CPU launch cost is roughly constant per step (~10 ms under Nsight, ~7.6 ms without), so it hides better behind longer GPU steps (c=32) but still dominates here.
- Nsight's own overhead inflates the eager step (10.06 ms vs 7.59 ms unprofiled), so the unprofiled numbers are the ones to cite.

## Caveats and open questions
- `--enforce-eager` also disables `torch.compile` fusion, so "graphs off" means graphs *and* compiler off (eager launched 389 vs 331 kernels per step at c=1). Isolating graphs alone would need `cudagraph_mode` set to NONE with compilation left on.
- Single short runs (c=1: 4 requests, c=32: 64 requests); each kernel-time figure is a total over the captured window. Steps are counted by sampling kernels, so prompt steps are included in per-step averages.
- The profiler adds overhead; the c=1 profile-based total is +4% over unprofiled TPOT.
- The c=1 trace shows prompt iterations scheduling only 8 tokens (`execute_context_1(8)`), which may mean the benchmark's repeated random prompts hit vLLM's prefix cache. Not checked; if true it affects how prefill cost reads in Days 1-3.
- Raw traces (~5-14 MB each) are not committed.

## Next
- Check the prefix-cache hit rate in the server log and rerun with `--no-enable-prefix-caching`.
- Isolate CUDA graphs from compilation; repeat each point several times.
- Try a larger model, where bigger matmuls should sit closer to peak bandwidth, to test the "bigger kernels run closer to peak" finding.
