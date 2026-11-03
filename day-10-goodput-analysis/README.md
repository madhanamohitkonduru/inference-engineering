# Day 10: Percentiles, goodput and capacity under an SLO

Part A re-analyzes the Day 9 results (no GPU). Part B is a new concurrency sweep on the GPU, analyzed for percentiles, goodput and capacity.
Model: Qwen3-0.6B (BF16), one RTX 4090, vLLM 0.32.0, `--max-model-len 4096`. *Finding* = measured; *hypothesis* = not tested.
**SLO used throughout:** p99 TTFT <= 200 ms and p99 TPOT <= 10 ms. "Meets the SLO" for a single request means its own TTFT <= 200 ms **and** TPOT <= 10 ms (this is vLLM's `--goodput ttft:200 tpot:10`);
a concurrency level "passes" when at least 99% of its requests meet both limits.

## Summary (for a team deciding how many GPUs to provision)
On one RTX 4090 serving Qwen3-0.6B (512-token prompts, 128-token outputs, prefix caching off), an SLO of p99 TTFT <= 200 ms and p99 TPOT <= 10 ms holds up to 16 concurrent users, where 99.2% of requests meet both limits and goodput is 23.9 requests/s (about 3,000 output tokens/s); beyond that, goodput falls even as throughput rises (at 32 users the server completes 35.6 requests/s but only 43.6% are on time, so goodput drops to 15.6).
TTFT is what breaks first, because requests queue for prefill, and TPOT does not cross 10 ms until 64 users. Capacity is very sensitive to the target: p99 TTFT <= 100 ms cuts it to 2 users, and <= 400 ms raises it to 32, after which the TPOT limit takes over.
To plan, divide peak requests per second by about 24 and round up (100 requests/s needs 5 GPUs, from 100 / 23.9 = 4.2, before any headroom). This is one pod, a 0.6B model and fixed-length random prompts, and the c=16 result is only 0.2 points above the 99% line, so re-measure with your real model, traffic and hardware before buying.

## Part A: Day 9 results in pandas and DuckDB
- `analyze_day9.py` loads the 27 files in `day-09-speculative-decoding/results/` (1,300 per-request rows) into one DataFrame; the config, run, workload and concurrency come from the file names.
- **Check against a known answer:** the pooled means reproduce the Day 9 README (base copy c=1 2.140 ms, base copy c=32 3.991, base free c=32 2.880, ngram k=4 copy c=1 0.868, k=16 copy c=1 0.340).
- The same aggregation in DuckDB (`queries.sql`, `quantile_cont`) matches the pandas table on every row.
- Each file has 20 requests, so pooled p99 uses 40 requests (20 for the single-run async-off free c=1 row): **a hint, not a result** (it is roughly the second-slowest request).

| c=1, free writing | n | mean | p50 | p95 | p99 |
|---|---|---|---|---|---|
| base | 40 | 2.135 | 2.133 | 2.139 | 2.190 |
| ngram k=4 | 40 | 1.672 | 1.452 | 2.949 | 3.472 |

*Finding (n=40, p99 is a hint):* speculation improves the mean TPOT on free writing but makes the tail worse than the baseline (p99 3.47 vs 2.19 ms; p99 is 2.4x its own p50, while the baseline's is 1.03x). At c=32 the same pattern holds (mean 2.95 vs 2.88, p99 5.35 vs 3.02).
A p99 TPOT SLO would therefore see speculation on free writing as worse than the baseline, even though the mean says it is better.

## Part B: concurrency sweep
Workload: `vllm bench serve` random dataset, 512 input / 128 output tokens, `openai-chat` endpoint, `--save-detailed`. **Prompts per point:** 500 for c = 1 to 32, 640 at c=64, 1,280 at c=128, 2,560 at c=256 (at least 10x the concurrency above c=50). Every point was run **twice** and the two runs are **pooled** (so 1,000 to 5,120 requests per point).
Prefix caching off for the main sweep (`sweep_cacheoff/`); a second set at c=1 and c=32 with caching on (`sweep_cacheon/`). Raw per-request data is gzipped JSON; `sweep_requests.parquet` is the combined per-request table. Script: `scripts/sweep_day10.sh`.
Per-request TPOT is computed as (latency - TTFT) / (output tokens - 1). My first version divided the whole latency by the token count and gave a mean of 5.07 ms against vLLM's 3.98; reconciling with vLLM's own summary caught it.

### Latency vs throughput (prefix caching off, two runs pooled)
| c | Prompts / run | Throughput (tok/s) | TTFT p50 | TTFT p95 | TTFT p99 | TPOT p50 | TPOT p95 | TPOT p99 |
|---|---|---|---|---|---|---|---|---|
| 1 | 500 | 302 | 42.5 | 55.0 | 61.2 | 2.93 | 3.37 | 3.51 |
| 2 | 500 | 558 | 66.7 | 78.5 | 86.9 | 3.01 | 3.53 | 3.74 |
| 4 | 500 | 1,062 | 75.2 | 90.2 | 104.8 | 3.08 | 3.64 | 3.78 |
| 8 | 500 | 1,905 | 106.0 | 127.1 | 140.4 | 3.28 | 3.81 | 4.16 |
| 16 | 500 | 3,082 | 145.7 | 178.9 | **195.8** | 3.97 | 4.39 | 4.65 |
| 32 | 500 | 4,561 | 209.5 | 294.6 | 307.9 | 5.24 | 5.96 | 6.18 |
| 64 | 640 | 5,798 | 302.7 | 505.7 | 609.0 | 8.64 | 9.80 | **10.17** |
| 128 | 1,280 | 7,068 | 208.1 | 795.6 | 1,152.4 | 16.16 | 16.77 | 17.01 |
| 256 | 2,560 | 6,652 | 307.6 | 1,593.2 | 2,628.3 | 36.13 | 37.79 | 38.43 |

(TTFT and TPOT in ms.) Chart: `latency_vs_throughput.png` (throughput against p99 TPOT and p99 TTFT, SLO limits dashed).
*Finding:* throughput grows about 1.8x per doubling up to c=8, then fades (1.6x, 1.5x, 1.3x, 1.2x), and **falls** from c=128 to c=256 while latency keeps climbing. The knee in throughput is around c=32 to 64; the SLO breaks earlier, between c=16 and c=32.

### Goodput (SLO: TTFT <= 200 ms and TPOT <= 10 ms)
| c | Throughput (req/s) | Goodput (req/s) | % of requests meeting both limits |
|---|---|---|---|
| 1 | 2.36 | 2.36 | 100.00 |
| 2 | 4.36 | 4.36 | 100.00 |
| 4 | 8.30 | 8.30 | 100.00 |
| 8 | 14.89 | 14.89 | 100.00 |
| 16 | 24.08 | **23.89** | **99.20** |
| 32 | 35.63 | 15.57 | 43.60 |
| 64 | 45.29 | 8.92 | 19.53 |
| 128 | 55.22 | 0.04 | 0.08 |
| 256 | 51.97 | 0.00 | 0.00 |

My goodput, computed from the per-request data (`n_ok / duration`, averaged over the two runs), **equals vLLM's reported `request_goodput` in every row**.
*Finding:* goodput peaks at c=16. From c=16 to c=32, throughput goes **up 48%** while goodput goes **down 35%**: more work in flight, but most of it arrives too late to count. At c=32 the failures are TTFT (p99 TPOT is 6.2 ms against a 10 ms limit).

### Capacity (highest concurrency where at least 99% of requests meet both limits)
| TTFT limit (TPOT <= 10 ms) | Capacity |
|---|---|
| 50 ms | none (c=1 already has p99 TTFT 61 ms) |
| 100 ms | c=2 |
| 200 ms | c=16 |
| 400 ms | c=32 |
| 1,000 ms | c=32 |

*Finding:* tightening TTFT from 200 to 100 ms cuts capacity from 16 to 2. Loosening to 400 ms doubles it to 32; loosening further gains nothing, because the p99 TPOT at c=64 is 10.17 ms, just over the limit.
(*Inference:* I did not print the exact fraction at c=64, but a p99 above 10 ms means fewer than 99% meet the TPOT limit, so TPOT is what stops c=64 once TTFT is relaxed.)
**Fragility:** at c=16 the two runs gave p99 TTFT of 186.1 and 220.1 ms (pooled 195.8), so passing there is borderline. c=8 passes comfortably in both runs (p99 TTFT 140 ms).

### Is the TTFT tail just the start-up burst? (first c requests vs the rest, TTFT in ms, two runs pooled)
| c | First wave p50 / p99 | The rest p50 / p99 |
|---|---|---|
| 32 | 237 / 311 | 207 / 305 |
| 64 | 422 / 672 | 294 / 519 |
| 128 | 796 / 1,303 | 199 / 435 |
| 256 | 1,591 / 2,945 | 294 / 942 |

*Finding:* at c=128 and 256 the pooled tail is mostly the first wave (its median is 4x to 5x the rest at c=128). At c=32 and 64 later requests queue about as much as the first wave, so the capacity result is **steady-state** behaviour, not a start-up artifact.
(This assumes the order of the per-request lists is the order requests were sent; I did not check `start_times`.)
*Unexplained:* the "rest" median TTFT at c=128 (199 ms) is lower than at c=64 (294 ms).

### Prefix caching off vs on, same pod (carry-over from Day 1)
| Same pod, two runs each | Caching off | Caching on |
|---|---|---|
| c=1 throughput | 302.3 tok/s | 298.2 tok/s |
| c=32 throughput | 4,560 tok/s | 4,562 tok/s |
| c=32 p99 TTFT (pooled) | 307.9 ms | 318.1 ms |
| c=32 % meeting the SLO | 43.6% | 44.4% |

*Finding:* no measurable effect on this workload (the differences are inside the run-to-run spread; the server log reported `enable_prefix_caching=True` and a 2.7% hit rate, as expected for random prompts that share no prefix). This closes the carry-over question: caching off is not why these numbers differ from Day 1.
Against the Day 1 reference (4090, caching on), this pod is 26% lower at c=1 (302 vs 410.8 tok/s) and 7% lower at c=32 (4,560 vs 4,927.5).
*Hypothesis (not tested):* this pod's CPU is slower (Intel Xeon Gold 6133 at 2.5 GHz, see `pod_info.txt`), which would hurt c=1, where steps are CPU-bound, most. The CPUs of the earlier pods were not recorded, so I cannot compare; next time I will record them at the start.

## Prediction vs measurement
| Prediction (before analysing) | Predicted | Measured | Result |
|---|---|---|---|
| Which limit breaks first as c rises | TTFT (queueing grows with c) | TTFT fails at c=32; TPOT holds until c=64 | correct |
| Highest passing concurrency | 32 (from Day 1's p99 TTFT of about 172 ms at c=32, 100 prompts) | 16 (p99 TTFT 307.9 ms at c=32) | wrong: 32 is 2x too high |
| Share of requests meeting the SLO at the predicted capacity | "all are good" | 43.6% at c=32 | wrong |
| TTFT of the last request in the first wave, c=128 / c=256 (prompt tokens / 2,048 per step x 12 ms) | 384 ms / 768 ms | p99 of the first wave 1,303 ms / 2,945 ms | about 3.4x and 3.8x too low |
*Why the first-wave estimate was low (hypothesis, not tested):* the 12 ms per step counted only prefill; steps also carry decode work for requests already started, and later prompt tokens cost more per token. Day 7's straight-line prefill model was also 40 to 60% too low.
*Why the capacity estimate was high (hypothesis):* Day 1's 172 ms came from 100 prompts, and a p99 from 100 requests is close to the single slowest; the 500-prompt runs see more queueing, and Day 1 also ran on a different pod.

## Self-check answers
1. **Why can a service with a great average still have unhappy users?** The average blends fast and slow requests: a few very slow ones barely move it, but the users who got them are unhappy. Example: nine requests at 2 to 5 ms and one at 40 ms average 6.7 ms (median 3 ms), and nobody got 6.7. Percentiles show what typical and worst-case users get.
2. **Why is p99 from 100 requests unreliable?** It is decided by the one or two slowest requests, so a single hiccup moves it. In the Day 9 files, p99 of 20 requests was exactly the first (slowest) request. At c=16 here, 500-request runs gave p99 TTFT 186 and 220 ms.
3. **Can throughput go up while goodput goes down?** Yes, and it happened here: from c=16 to c=32, throughput rose from 24.1 to 35.6 requests/s while goodput fell from 23.9 to 15.6, because only 43.6% of requests met the SLO.

## What I did not do
- Closed-loop load only (a fixed number of requests in flight, all asking for 128 output tokens); no open-loop arrival rate and no real traffic shapes.
- One model (0.6B), one GPU, one pod, random prompts of fixed length; results will differ for larger models or longer prompts.
- The Day 1 sweep was not rerun in its original form; the new sweep replaces it and adds per-request TTFT/TPOT.
- The upstream PR hunt (issues labelled "good first issue" or documentation in vLLM and guidellm): **not started**.

## Files
`analyze_sweep.py` (loads the sweep, builds the tables, chart, goodput and capacity), `analyze_day9.py`, `queries.sql` (Day 9 percentiles), `queries_sweep.sql` (sweep percentiles and % meeting the SLO),
`latency_vs_throughput.{csv,png}`, `goodput_table.csv`, `day9_requests.parquet`, `sweep_requests.parquet`, `sweep_cacheoff/`, `sweep_cacheon/` (raw benchmark output), `pod_info.txt`.
