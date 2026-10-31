# Day 7: chunked prefill

Server: vLLM 0.32.0, Qwen3-0.6B, RTX 4090, `serve.sh --no-enable-prefix-caching` (so the long prompt cannot be skipped by the cache), `--max-model-len 4096`.
Chunked prefill is **on by default** (`enable_chunked_prefill=True` in the engine config). Script: [`chunked_prefill_test.py`](chunked_prefill_test.py). Raw output: [`results/raw_output.txt`](results/raw_output.txt).

## 1. Self-check answers (my own words)
1. **Why does a long prompt stall other users?** Prefill is done in the same step as everyone's decode; if a big share of the compute goes to the long prompt, the step takes much longer and the decoding users get no new token until it ends.
2. **What happens to the long request's own TTFT as the budget shrinks?** It needs more steps (steps ~ prompt tokens / chunk size), so its first token arrives later. Its first token only appears once the last chunk is done. (Correction from review: tokens / budget is the number of steps, not the time; TTFT ~ steps x time per step, and the chunk is the budget minus the tokens taken by running requests.)
3. **Why are running requests scheduled before waiting ones?** Their cost is tiny (1 token each) and a skipped running request freezes a user who is mid-answer; the waiting request takes whatever budget is left. (Correction from review: running requests do not have to finish first; each gets one token every step while the chunk rides along.)

## 2. What the default budget is
`max_num_batched_tokens` is not a fixed default. `arg_utils.py` (lines ~2915-2985) picks it from a table by device memory and usage context: a device under 70 GB (this 24 GB 4090) running `vllm serve` gets **2,048** (and `max_num_seqs` 256). The default run behaved exactly like the explicit 2,048 run (two heavy steps, same stall), which supports this. So **Days 1-6 also ran with a 2,048 budget**.

## 3. Predictions vs measurements
Prefill speed from Day 6 (cold TTFT ~28 ms minus warm ~16 ms = ~12 ms for 2,000 tokens) = **6 ms per 1,000 tokens**. Reference predictions (Claude's; one decode step ~2.5 ms added; intruder TTFT adds ~13 ms fixed overhead):

| Budget | Chunks predicted | Chunks measured (victim 0's heavy steps) | Stall predicted | Stall measured (2 runs) | Gap | Intruder TTFT predicted | Measured (2 runs) | Gap |
|---|---|---|---|---|---|---|---|---|
| 8,192 | 1 | 1 | 23.5 ms | 34.8 / 36.5 ms | +52% | ~37 ms | 44.9 / 46.1 ms | +23% |
| 2,048 | 2 | 2 | 14.8 ms | 20.8 / 20.2 ms | +39% | ~39 ms | 47.3 / 47.1 ms | +21% |
| 512 | 7 | 6 visible (>5.4 ms threshold) | 5.6 ms | 8.9 / 9.2 ms | +62% | ~52 ms | 54.5 / 56.2 ms | +6% |

(Percentages use the average of the two runs.) The order of everything was predicted correctly (smaller budget: smaller stall, slower long request); the sizes were underestimated because the straight-line model ignored the effects in section 5.

## 4. Results

| Budget | Normal gap (baseline median) | Median gap with intruder | Worst gap (stall) | Intruder TTFT | Heavy steps seen by victim 0 (ms) |
|---|---|---|---|---|---|
| 8,192 | 2.68 | 2.73 / 2.71 | 34.8 / 36.5 | 44.9 / 46.1 | 6.3 (admission), 34.8 // 6.1, 36.5 |
| 2,048 (explicit) | 2.75 | 2.77 / 2.75 | 20.8 / 20.2 | 47.3 / 47.1 | 5.6, 16.9, 20.8 // 16.8, 20.2 |
| 2,048 (default, 1 run) | 2.73 | 2.77 | 19.8 | 46.7 | 5.5, 16.4, 19.8 |
| 512 | 2.69 | 2.70 / 2.68 | 8.9 / 9.2 | 54.5 / 56.2 | 6.2, 6.5, 7.0, 7.3, 8.1, 8.9 // 6.1, 6.5, 6.9, 7.6, 7.7, 9.2 |

All values in ms. Baseline (no intruder) worst gaps were 3.4-3.6 ms at every budget, so the stalls are far above the noise.

- **Finding:** the stall shrinks with the budget (~36 ms -> ~20 ms -> ~9 ms, about -75% from 8,192 to 512) and the long request's TTFT grows (~45 -> ~47 -> ~55 ms, about +20%). The median and p99 gaps barely change because only a handful of steps are disturbed.
- **Finding:** going from 8,192 to 2,048 cut the worst stall ~43% and cost ~2 ms of TTFT. The summed step times for the prompt were ~36 ms (one step), ~37 ms (two steps) and ~44 ms for the six visible steps at 512 (a seventh was below the threshold), so many small steps do cost extra total time.
- **Finding:** every victim sees the same stalled steps (all five largest gaps were identical in every run): a step is shared, so everyone in it waits for the whole step.

## 5. Why the sizes were underestimated
- **Finding:** the same-size chunk gets slower as more of the prompt has been done. At budget 512 every chunk has the same size (506 tokens), yet the heavy steps climb 6.2 -> 6.5 -> 7.0 -> 7.3 -> 8.1 -> 8.9 ms (and 6.1 -> ... -> 9.2 ms in run 2).
- **Finding:** per 1,000 prompt tokens (step time minus a 2.7 ms normal step): first chunk at 2,048 (2,042 tokens, no earlier text) ~6.7 ms; second chunk (1,458 tokens, 2,042 earlier) ~11.7 ms; the whole 3,506 tokens in one step ~9.4 ms.
- **Hypothesis (not tested):** attention has to look back over all earlier tokens, so a token costs more when more text precedes it; my straight-line model missed that.
- **Hypothesis (not tested):** steps over 512 tokens cannot use a full CUDA graph (the log shows graphs captured up to 512), which might add CPU launch time to the larger steps. The 512-budget chunks show no sign of a graph speed-up (first chunk ~6.8 ms per 1,000 tokens, the same as the first 2,048 chunk), but this was not isolated.

## 6. Chat vs batch (my answer)
- **Chat product: 512.** It gives the smallest stall for users watching text stream (9 ms vs 36 ms). Cost: the long prompt's TTFT is ~10 ms slower (*finding*).
- **Overnight batch: 8,192.** Nobody is waiting on an individual token. It needs the fewest steps for the same work: the prefill took ~36 ms of step time against ~44+ ms at 512 (*finding*). **Hypothesis:** the saving comes from fewer steps, each paying the weight read from memory (Day 3) and per-step overhead.
- Review notes: a 36 ms stall is probably too short for a person to notice (*hypothesis*, not tested), so for chat the default 2,048 is also defensible; a bigger budget also needs more working memory per step (*hypothesis*, not measured).

## 7. Reading the scheduler (`vllm/v1/core/sched/scheduler.py`, v0.32.0)
- Waiting request cap: line 1169 `num_new_tokens = min(num_new_tokens, request_token_budget)`. If chunked prefill is **disabled** and the prompt does not fit, lines ~1160-1166 `break` and the request waits whole.
- Running requests (lines ~702-710): `num_new_tokens = (all tokens the request has) - (tokens already computed)`, capped by the budget. This gives 1 for a decoding request and the rest of the prompt for a half-prefilled one, so a chunked prompt is an ordinary running request on its next step (`is_prefill_chunk`, line 638).
- `long_prefill_token_threshold` (lines ~640-654, default 0 = off in `config/scheduler.py`): an extra cap on tokens per step for one long prefill, applied only when more than one request is around; an adaptive version floors it at a fair share of the budget (default off).

Three sentences:
1. When a waiting prompt is larger than the remaining budget, the scheduler gives it only the remaining budget (a chunk), `min(num_new_tokens, request_token_budget)`, as long as chunked prefill is enabled.
2. On the next step the request is in the running list, and the scheduler computes how many tokens it still needs as all its tokens minus the tokens already computed, capped by the budget, so it takes the next chunk.
3. `long_prefill_token_threshold` adds an extra cap on how many tokens one long prefill may take in a step; it is off by default (value 0).

## Caveats
- One pod, two intruder runs per budget (one for the default), one baseline per budget; one intruder size (3,500 tokens) and six victims.
- Timestamps are taken on the client with Python threads; they agree with each other to about 1 ms across runs but are not server-side measurements.
- The seventh step at budget 512 was not visible with the 5.4 ms threshold, so the 512 step-time sum is a lower bound.

## Next
- Quantization (FP8 vs BF16 plus a 4-bit scheme).
- Carry-overs: rerun the Day 1 sweep with prefix caching off; pandas/DuckDB percentile analysis of Day 1 data.
