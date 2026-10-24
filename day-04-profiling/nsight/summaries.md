# Nsight Systems summaries (output of `scripts/nsys_gaps.py`, one report per capture)

Server wrapped in `nsys profile ... --capture-range=cudaProfilerApi --capture-range-end=repeat --cuda-graph-trace=node`
with `--profiler-config.profiler cuda`. c=1: 4 requests; c=32: 64 requests. Same random 512-in/128-out workload.

## Graphs ON (default)
**c=1** (`graphs.1`)
- GPU activities: 186,348; window: 1,215.4 ms; busy: 1,164.8 ms (95.8%); idle: 50.6 ms
- Gaps: 185,410; median 0.06 us; p99 2.4 us; gaps > 20 us: 163 totalling 23.1 ms
- Steps counted (kernels matching 'temperature'): 512
- Per step: window 2.374 ms; busy 2.275 ms; idle 0.099 ms; kernel launches per step: 331

**c=32** (`graphs.2`)
- GPU activities: 132,971; window: 1,436.8 ms; busy: 1,372.3 ms (95.5%); idle: 64.5 ms
- Gaps: 126,956; median 0.06 us; p99 2.2 us; gaps > 20 us: 118 totalling 12.4 ms
- Steps counted: 269
- Per step: window 5.341 ms; busy 5.102 ms; idle 0.240 ms; kernel launches per step: 436

## Graphs OFF (`--enforce-eager`)
**c=1** (`eager.1`)
- GPU activities: 215,536; window: 5,151.2 ms; busy: 1,184.6 ms (23.0%); idle: 3,966.6 ms
- Gaps: 215,163; median 13.79 us; p99 74.9 us; gaps > 20 us: 51,409 totalling 2,341.1 ms
- Steps counted: 512
- Per step: window 10.061 ms; busy 2.314 ms; idle 7.747 ms; kernel launches per step: 389

**c=32** (`eager.2`)
- GPU activities: 147,381; window: 3,079.4 ms; busy: 1,412.5 ms (45.9%); idle: 1,666.9 ms
- Gaps: 147,153; median 1.86 us; p99 63.6 us; gaps > 20 us: 19,483 totalling 907.7 ms
- Steps counted: 267
- Per step: window 11.533 ms; busy 5.290 ms; idle 6.243 ms; kernel launches per step: 494

## Plain benchmark, no profiler (`scripts/bench_quick.sh`, 100 prompts, c=1 and c=32)
| | Graphs ON | Graphs OFF (`--enforce-eager`) | Slowdown |
|---|---|---|---|
| c=1 output tok/s | 428.7 | 129.9 | 3.3x |
| c=1 mean TPOT | 2.26 ms | 7.59 ms | 3.4x |
| c=32 output tok/s | 5,635.8 | 2,878.1 | 2.0x |
| c=32 mean TPOT | 4.71 ms | 8.59 ms | 1.8x |
