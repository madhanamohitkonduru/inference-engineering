# Day 4: where does decode time go? (in progress)

Goal: explain the gap between the Day 3 memory-bandwidth roofline and the measured decode speed of Qwen3-0.6B on an RTX 4090.
Status: PyTorch-profiler part done; Nsight Systems and CUDA-graphs-vs-eager still to do.

## Setup
- RTX 4090 (24 GB), driver 595.91.07, vLLM 0.32.0 (same as Days 1 and 3). Environment rebuilt with [`scripts/setup.sh`](../scripts/setup.sh).
- Sanity check against Day 1 (`scripts/bench_quick.sh`): c=1 428.7 tok/s, TPOT 2.26 ms (Day 1: 410.8, 2.25); c=32 5,635.8 tok/s, TPOT 4.71 ms (Day 1: 4,927.5, 4.91). TPOT agrees within 4%; the 100-prompt c=32 throughput is noisy.

## Method (PyTorch profiler)
```bash
bash scripts/serve.sh --profiler-config '{"profiler":"torch","torch_profiler_dir":"/root/prof/torch","torch_profiler_with_stack":false,"ignore_frontend":true,"max_iterations":300}'
# warm-up (not recorded): 30 prompts, c=8; then recorded runs with --profile:
vllm bench serve ... --num-prompts 4  --max-concurrency 1  --profile   # c=1
vllm bench serve ... --num-prompts 64 --max-concurrency 32 --profile   # c=32
python3 scripts/trace_top_kernels.py <trace.json.gz>
```
Raw summaries: [`results/torch_profiler_c1_summary.md`](results/torch_profiler_c1_summary.md), [`results/torch_profiler_c32_summary.md`](results/torch_profiler_c32_summary.md).

## Findings
| | c=1 | c=32 |
|---|---|---|
| GPU busy (share of window) | 95.3% | 95.4% |
| Attention | 18.8% of kernel time | 49.5% |
| GEMM | 65.3% | 38.7% |
| Other | 15.9% | 11.8% |

1. **The GPU is ~95% busy at both concurrencies**; gaps between kernels are tiny (median 0.1 us). With CUDA graphs on, idle time is not the main cause of the Day 3 gap; kernel execution time is.
2. **c=1 step budget** (293 decode steps; totals divided by 293, approximate): layer GEMVs ~1.16 ms, LM head ~0.33 ms, attention ~0.4 ms, other small kernels ~0.37 ms; total ~2.25 ms, matching measured TPOT (2.26 ms).
3. **Bigger kernels get closer to peak bandwidth** (see assumptions below): LM head ~94%, layer GEMVs ~75%, attention ~17% at c=1; attention at c=32 (~2.1 GB of KV per step, ~2.8 ms) reaches ~75%.
4. **Batching win visible in the trace:** GEMM time per decode step is ~1.4 ms at c=32 versus ~1.5 ms at c=1, while serving 32 sequences.
5. At c=1 the layer matmuls are matrix-vector kernels (`gemvx`); at c=32 they are tensor-core GEMMs (`cutlass`/`ampere` gemm). Two large-tile GEMM kernels with few calls (704, 868) look like prompt processing (~14% of kernel time at c=32; inferred from call counts, not labelled in the trace).

## Assumptions and caveats (not yet verified)
- Bytes per kernel are computed from Qwen3-0.6B's architecture as I recall it (hidden 1024, intermediate 3072, 16 heads / 8 KV heads, head_dim 128, vocab 151,936, 28 layers), not read from `config.json` yet.
- **Day 3 correction pending:** the Day 3 roofline counted 1.503 GB of weights (checkpoint size). The profile suggests ~1.19 GB is read per step (881 MB layers + 311 MB LM head; the embedding table is a row lookup). That would raise the c=1 ceiling to ~800 tok/s and enlarge the gap to ~45%. To be confirmed against `config.json` and then Day 3 updated.
- Short single recordings (300 iterations cap), profiler enabled; step counts at c=32 (~245) inferred from kernel call counts, so per-step times are approximate.
- Whether prefix caching served part of the benchmark prompts (benchmark reuses the same random prompts) is unchecked; it would change how prefill cost is read in Days 1-3.

## Still to do
- Nsight Systems capture at c=1 and c=32; CUDA graphs on vs `--enforce-eager`; idle gap per decode step.
- Re-check Day 3 roofline with corrected weight bytes.
