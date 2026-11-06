# Day 11: Correctness harness, part 1

Goal: replace "do the outputs match?" with a measurement that separates real model changes from harmless numerical noise.
Method: **teacher forcing**. Both configs read the same fixed text and I compare their next-token logprobs at every position, so they cannot drift apart.
Model: Qwen3-0.6B, vLLM 0.32.0, RTX 4090, `--no-enable-prefix-caching --max-model-len 4096 --gpu-memory-utilization 0.85` for every server (confirmed from the server logs, `server_flags_check.txt`).
*Finding* = measured; *hypothesis* = not tested.

## Setup
- **Passages:** 50 passages of 512 tokens from the WikiText-2 raw **test** split (real Wikipedia text no model wrote), spread evenly over the test set; `passages.json` holds the token ids and source. Each passage gives 511 scored positions (the first token has no context), so **25,550 positions per capture**.
- **Capture:** `/v1/completions` with the prompt as token ids, `max_tokens=1`, `prompt_logprobs=20`. Per position I save the true next token's logprob, the top-1 id, and the top-20 [id, logprob] list. Format check: the raw response has 21 entries per position (top 20 plus the true token if it is not in the top 20).
- **Pod:** AMD Ryzen 9 7950X (32 CPUs), RTX 4090 (24,564 MiB), driver 595.91.07, 2026-11-06 19:36 UTC (`pod_info.txt`). Day 10's pod had a Xeon Gold 6133.
- **Captures (one run each unless noted):** `bf16_c1_run1`, `bf16_c1_run2`, `bf16_c32` (same BF16 server, 32 requests in flight), `fp8_c1` (`--quantization fp8`), `fp8kv_c1` (`--quantization fp8 --kv-cache-dtype fp8`). Reference for every pair is `bf16_c1_run1`.
- **Metrics (`logprob_compare.py compare A B`):** mean |true-token logprob difference|; top-1 flip rate; approximate KL from the two top-20 lists (union of both lists; a token missing from one list gets that list's smallest top-20 probability; each side renormalized; this is an approximation); perplexity = exp(-mean true-token logprob).
- **Tests:** `test_logprob_compare.py` (no GPU): a file against itself reports exactly zero; more fake noise gives larger numbers on every metric. On real data the self-comparison also printed exact zeros.
- Note: the `--note` saved inside `bf16_c32.json.gz` has a typo (`--max-model-len4096`); the data and the server flags (checked in the logs) are unaffected.
- **c=32 here is not 32 sequences per step.** The passages are 512 tokens, so a 2,048-token step holds at most about 4 of them; it still changes the batch shape relative to c=1.

## Results (each pair vs `bf16_c1_run1`, 25,550 positions)
| Pair | Mean \|logprob diff\| | Max \|diff\| | Top-1 flip rate | Mean KL (top-20) | Perplexity (ref 26.8387) | Change |
|---|---|---|---|---|---|---|
| Control: BF16 run 2, c=1 | 0 | 0 | **0%** | 0 | 26.8387 | 0 |
| Batch shape: BF16 c=32 | 0.0298 | 0.54 | **1.76%** | 0.00083 | 26.8367 | -0.0020 |
| FP8 weights | 0.1563 | 5.10 | **7.71%** | 0.01887 | 27.2692 | +0.4304 (+1.6%) |
| FP8 weights + FP8 KV | 0.1929 | 6.28 | **9.33%** | 0.02752 | 27.4928 | +0.6541 (+2.4%) |
FP8 vs FP8+KV directly: flip 9.13%, mean diff 0.179, KL 0.0241, perplexity +0.22.
Table also in `pair_table.csv`.

- *Finding:* the control is **exactly zero**: the same config with the same batch shape reproduces every number.
- *Finding:* batch shape alone is **not** zero: the noise floor is a 1.76% flip rate, mean diff 0.030, KL 0.0008.
- *Finding:* FP8 is above that floor on every metric (flip 4.4x, mean diff 5.2x, KL 23x) and FP8+KV is above FP8 (flip +1.6 points, KL +46%, perplexity +0.22).
- *Finding:* the damage in perplexity terms is modest: +1.6% (FP8) and +2.4% (FP8+KV).
- Single run per config, one set of 50 passages: no confidence intervals. The c=32 and FP8 numbers could move on a rerun; BF16 c=1 is deterministic.

## Histogram (`logprob_diff_hist.png`, log-scale y)
| Pair | median \|diff\| | p99 | p99.9 | max | positions > 0.5 |
|---|---|---|---|---|---|
| Batch shape | 0.015 | 0.15 | 0.27 | 0.54 | 0.01% |
| FP8 | 0.092 | 0.93 | 1.76 | 5.10 | 5.6% |
| FP8 + KV | 0.110 | 1.17 | 2.11 | 6.28 | 9.2% |
*Finding:* batch-shape changes are tightly bunched near zero; FP8 shifts the whole distribution right and has a long thin tail.

## Most-changed positions under FP8 (`top10_changed_fp8.txt`)
- 3 large real drops of the true word's probability: " Williams" after "William Carlos" 57.8% to 0.4% (FP8's top-3 became " and"/" in" at 21.8% each, " has" 11.7%), "ugh" in "Tombaugh" 35% to 0.3%, " l" 34.7% to 1.6%.
- 1 large gain ("ceph" in "Stegoceph", 3.3% to 72.1%): changes go both ways.
- 2 moderate changes, and **4 where the true word is at 0.0% to 0.1% on both sides** (e.g. logprob -14.5 vs -9.7): the log scale exaggerates tiny probabilities, so a big logprob difference there means little.
- Candidate tokens are shown as vocabulary ids when the token did not appear in the passages (this machine had no tokenizer).
- *Hypothesis (10 examples, untested):* FP8 hurts recall of rare names more than common words (4 of the 10 involve names).

**Are flips near-ties?** Share of flips where BF16's top-1 and top-2 were within 5 percentage points (for reference, 18% of all positions are that close):
batch shape **96%** (median gap 0.2 points), FP8 **73%** (median 2.4), FP8+KV **67%** (median 2.9).
*Finding:* batch-shape flips are almost entirely coin-flip ties (rounding), while a quarter to a third of FP8 flips are not ties: FP8 changed the model's belief.

## Does the flip rate explain Day 8 and Day 9? (prediction (1-p)^100 for 100-token outputs, 50 outputs)
| Setup | Measured flip rate p | (1-p)^100 x 50 | Earlier result | p implied by that result |
|---|---|---|---|---|
| FP8 weights | 7.71% | 0.02 | Day 8: 3/50 identical | 2.8% |
| FP8 + KV | 9.33% | 0.00 | Day 8: 1/50 | 3.8% |
| Batch shape (BF16 c=32) | 1.76% | 8.5 | Day 9 speculation: 16/50 | 1.1% |
Count of passages with no flip in their first 100 positions: batch shape 5/50, FP8 0/50, FP8+KV 0/50.
- *Finding:* the simple formula **does not** reproduce Day 8's identical counts (it predicts essentially none, Day 8 saw 3 and 1).
- *Finding:* the flip rate does explain **when** outputs split: Day 8's median first divergence at token 8 to 9 implies p about 7.8% (1 - 0.5^(1/8.5)), matching the measured 7.7%.
- *Finding:* of Day 8's 3 identical FP8 outputs, one was a very short answer (29 words, "17 times 23"); the formula assumes every token is an equal, independent coin flip, which real prompts are not.
- *Finding:* batch-shape noise predicts about 8.5 of 50 identical; Day 9's speculation gave 16, so speculation's differences are no larger than ordinary batch-shape noise. That fits, but does not prove, the "rounding noise, not a bug" explanation for Day 9.
- *Hypothesis:* the model's own short, confident text has fewer near-ties than these hard Wikipedia passages, so the flip rate on Day 8 prompts is lower than 7.7%. Test: teacher-force the Day 8 outputs (needs a GPU).

## Prediction vs measurement
| Prediction (written before measuring) | Predicted | Measured | Result |
|---|---|---|---|
| Which pair is exactly zero | none ("control may differ slightly") | control exactly 0 | wrong |
| Ranking, smallest to largest | batch shape < FP8 < FP8+KV | same | correct |
| FP8 top-1 flip rate | more than 10% | 7.71% | too high (the Day 8 README guess of 4 to 6% was too low) |
| Identical outputs of 50 at your p (10%) | about 0.001 vs Day 8's 3/50 | at measured p: 0.02 | formula inconsistent with Day 8; p implied by Day 8 is about 3% |

## Self-check answers
1. **Why can the same model give slightly different logprobs at c=1 and c=32?** Floating-point addition is not associative: a different batch size makes the kernel add the same numbers in a different order, so rounding differs slightly even with identical weights. Here that gave a 1.76% flip rate.
2. **Why compare under teacher forcing instead of counting identical free-run outputs?** With the same fixed text, both configs see identical input at every position, so one flip cannot spread; I can measure size of change (logprobs, KL, perplexity) at each position. Counting identical outputs only says a flip happened somewhere in 100 tokens, not how large the change was.
3. **Why must a gate's threshold sit above the noise floor?** Below it, the gate fires on harmless changes such as batch size, and people stop trusting it. It must also stay below the damage we care about, or it misses real regressions.

## Code notes (Step 6)
In `vllm/v1/sample/sampler.py`, `compute_logprobs(logits)` returns `logits.log_softmax(dim=-1, dtype=torch.float32)`; `topk_topp_sampler.py` has the same call for the "processed_logprobs" mode.
So the logits are upcast to 32-bit floats and the log of the softmax is computed in float32, which adds almost no extra rounding at this last step. The precision decisions that matter are earlier: the logits come out of BF16 or FP8 matrix multiplies that are already rounded.
I did not separately locate the prompt-logprobs code path; I assume it uses the same function.

## Candidate gate thresholds (my reading of this data, not validated)
The noise floor (batch shape) is flip rate 1.8%, KL 0.0008, perplexity change about 0. The change I would want a gate to catch (FP8) is flip rate 7.7%, KL 0.019, perplexity +1.6%. A threshold between them, e.g. KL near 0.005 or perplexity +0.5% (checked together), would separate them here. With one noise run, one model and one text set, I would re-measure the noise floor on several runs before trusting any number.

## What I did not do
- One run per config (except BF16 c=1), 50 passages, one model, one GPU, one kind of text; no confidence intervals.
- KL uses only the top 20 candidates (approximation).
- No downstream-task evaluation: perplexity and flips are proxies for quality.
- Did not teacher-force the model's own Day 8/9 outputs (hypothesis above).
- Upstream PR hunt: **not started**.

## Files
`logprob_compare.py` (capture + compare), `test_logprob_compare.py`, `analyze_day11.py`, `passages.json`, `captures/*.json.gz` (raw), `pair_table.csv`, `logprob_diff_hist.png`, `top10_changed_fp8.txt`, `freerun_prediction.json`, `pod_info.txt`, `server_flags_check.txt`.
