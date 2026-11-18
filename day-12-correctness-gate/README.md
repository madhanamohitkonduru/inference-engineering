# Day 12: Correctness harness, part 2 (the gate)

Goal: `gate.py` takes a reference capture and a candidate capture and returns PASS or FAIL with a reason, using thresholds justified by data and a measured false-alarm rate.
Model: Qwen3-0.6B (BF16), vLLM 0.32.0, RTX 4090; same flags as Day 11 on every server (`--no-enable-prefix-caching --max-model-len 4096 --gpu-memory-utilization 0.85`, confirmed per server in `server_flags_check.txt`) plus the flag under test.
*Finding* = measured; *hypothesis* = not tested. Passages: the same 50 WikiText-2 test passages (25,550 scored positions) as Day 11. Reference for all comparisons: `new_bf16_c1`.

## What was built
- `passage_metrics.py`: per-passage flip rate, KL, perplexity and mean logprob difference; bootstrap and random-subset helpers.
- `bootstrap_day11.py`: bootstrap intervals and the small-subset test on Day 11's captures (no GPU).
- `gate.py REFERENCE CANDIDATE --thresholds gate_thresholds.json`: prints each metric, its 95% bootstrap interval (resampling passages), its threshold and PASS/FAIL, then a verdict. Exit code 0 = pass, 1 = fail, 2 = error (bad input or thresholds file), so CI can tell a failed gate from a broken one. Thresholds are read from JSON (a missing key gives a clear error). **The decision uses the point value; the interval is shown for information.**
- `test_gate.py` (no GPU): a capture against itself passes; tiny noise passes; large fake noise fails; the "min metrics failing" rule behaves as set; a thresholds file with a missing key gives a clear error and exit code 2; the gate's numbers equal Day 11's compare tool (this test was added after a bug, below).
- `candidate_table.py`, `scoreboard.py`: Step 6 tables.

## Setup of the GPU captures
Pod: AMD EPYC 7K62 (96 CPUs), RTX 4090, driver 595.112.07 (`pod_info.txt`; Day 11's pod was a Ryzen 9 7950X with driver 595.91.07).
| Label | Server flags added | Captured |
|---|---|---|
| `new_bf16_c1` | none | c=1 (new reference) |
| `noise_c2`, `noise_c8`, `noise_c16` | none (same server) | c=2, 8, 16 |
| `noise_eager_c1` | `--enforce-eager` | c=1 |
| `noise_chunk256_c1` | `--max-num-batched-tokens 256` | c=1 |
| `bad_rope_mild_c1` | `--hf-overrides '{"rope_parameters": {"rope_type": "default", "rope_theta": 500000}}'` | c=1 |
| `bad_rope_big_c1` | same with `rope_theta` 100000 | c=1 |
| `rope_flat_ignored_c1` | `--hf-overrides '{"rope_theta": 500000}'` (flat spelling, **ignored**, kept as evidence) | c=1 |
Day 11's `bf16_c32`, `fp8_c1`, `fp8kv_c1` were reused (see the cross-pod check).
The model's default `rope_theta` is **1,000,000** (read from its `config.json`; no rope scaling), so the two overrides are **0.5x** and **0.1x** of the default.

## Findings
- **Cross-pod check: exactly zero.** `new_bf16_c1` equals Day 11's `bf16_c1_run1` on every metric (flip rate 0%, KL 0) even though the CPU, driver and physical GPU differ. So Day 11's FP8 and FP8+KV captures were reused unchanged.
- **The first rope override was silently ignored.** `{"rope_theta": 500000}` was accepted with no error or warning and the run was exactly zero versus the reference; vLLM's config object showed `rope_parameters.rope_theta` still 1,000,000 (the flat key only added an unused attribute). The nested `rope_parameters` spelling worked (value became 500,000). Lesson: after any config override, check that the capture differs from the reference.
- **Noise sources are not all alike.** c=2 gave exactly zero; c=8, c=16 and c=32 gave 1.48%, 1.53%, 1.76% flip rate; chunk-256 1.87%; eager (CUDA graphs off) **2.06%**, the largest.
- **Mild rope damage is real and ladder-like:** rope 0.5x 5.06%, FP8 7.71%, FP8+KV 9.33%, rope 0.1x 12.93% flip rate.
- **Bug caught by running the real data:** my first gate compared a fraction (0.077) against a threshold in percent (4.0), so FP8's flip rate passed. The fake-data tests missed it because other metrics still failed; a test comparing the gate's numbers with Day 11's tool now covers it.

### Candidate table (vs `new_bf16_c1`, 95% bootstrap interval over the 50 passages in brackets)
| Kind | Candidate | Flip rate % | Mean KL | Perplexity change % |
|---|---|---|---|---|
| noise | c=2 | 0 | 0 | 0 |
| noise | c=8 | 1.48 [1.19, 1.77] | 0.00066 [0.00056, 0.00076] | -0.02 [-0.08, +0.04] |
| noise | c=16 | 1.53 [1.29, 1.79] | 0.00071 [0.00061, 0.00080] | -0.02 [-0.09, +0.05] |
| noise | c=32 (Day 11) | 1.76 [1.56, 1.96] | 0.00083 [0.00077, 0.00087] | -0.01 [-0.07, +0.06] |
| noise | chunk-256 | 1.87 [1.69, 2.06] | 0.00091 [0.00089, 0.00093] | +0.01 [-0.07, +0.08] |
| noise | eager | **2.06 [1.89, 2.26]** | **0.00105 [0.00103, 0.00107]** | -0.06 [-0.12, +0.005] |
| bad | rope 0.5x (mild) | **5.06 [4.68, 5.40]** | **0.00831 [0.00791, 0.00873]** | **+0.64 [+0.37, +0.91]** |
| bad | FP8 (Day 11) | 7.71 [7.35, 8.07] | 0.01887 [0.01798, 0.01978] | +1.60 [+1.27, +1.90] |
| bad | FP8 + KV (Day 11) | 9.33 [8.91, 9.75] | 0.02752 [0.02640, 0.02858] | +2.44 [+2.02, +2.81] |
| bad | rope 0.1x (big) | 12.93 [12.37, 13.45] | 0.05624 [0.05386, 0.05887] | +4.97 [+4.31, +5.71] |
(`candidate_table.csv`.) On every metric there is a gap between the highest noise interval end (flip 2.26%, KL 0.00107, perplexity +0.084%) and the lowest bad interval end (flip 4.68%, KL 0.00791, perplexity +0.37%).

## Thresholds (`gate_thresholds.json`) and why
Rule: a metric fails when its point value is **above** its threshold (perplexity is one-sided: only a rise counts); the gate **fails when at least 2 of 3** metrics fail.
| Metric | Threshold | Largest noise (interval end) | Mildest bad change (interval low end) | Margin reasoning |
|---|---|---|---|---|
| Flip rate | **3.5%** | 2.06% (2.26%) | 4.68% | 1.7x the largest noise, about half-way through the gap; I first considered 5% (more margin) but it sits inside the mild-rope interval (4.68 to 5.40), so another text set could miss mild rope |
| Mean KL | **0.004** | 0.00105 (0.00107) | 0.00791 | 3.7x the largest noise, half of the mild-rope low end; a 75%-of-the-gap line (about 0.006) was rejected as leaving only 1.3x under the mild damage |
| Perplexity change | **+0.2%** | +0.084 | +0.37 | 2.4x the largest noise, about half of the mild-rope low end; the Day 11 candidate of +0.5% lies inside the mild-rope interval, so it would be a fragile line |
**Margin choice (judgment, not measured):** I picked roughly half-way (in ratio terms) so that unseen noise sources have room above and mild damage has room below. Only six noise sources were measured, so "the largest noise" is a lower bound on the real maximum.
**Rule choice:** "at least 2 of 3" trades fewer false alarms from one twitchy metric (perplexity is the noisiest) against missing a regression that only one metric shows. On this data the rule changes almost nothing (see below), because the three metrics move together.

## Scoreboard (point values, `scoreboard.csv`)
| Kind | Candidate | Flip | KL | Perplexity | Metrics failed | Verdict (2 of 3) |
|---|---|---|---|---|---|---|
| noise | c=2 / 8 / 16 / 32 / chunk-256 / eager | 0 / 1.48 / 1.53 / 1.76 / 1.87 / 2.06 | 0 to 0.00105 | -0.06 to +0.008 | 0 of 3 | **PASS** (all six) |
| bad | rope 0.5x (mild) | 5.06 | 0.00831 | +0.64 | 3 of 3 | **FAIL** |
| bad | FP8 | 7.71 | 0.01887 | +1.60 | 3 of 3 | **FAIL** |
| bad | FP8 + KV | 9.33 | 0.02752 | +2.44 | 3 of 3 | **FAIL** |
| bad | rope 0.1x (big) | 12.93 | 0.05624 | +4.97 | 3 of 3 | **FAIL** |
*Finding:* every noise row passes and every bad change fails, including the mild rope change and FP8.

## False-positive and detection rates by number of passages (1,000 random subsets each, `subset_rates.csv`)
% of subsets on which the gate FAILS the candidate. Subsets of 10 and 25 are drawn without replacement; "50" is a bootstrap resample (with replacement), the only way to vary a 50-passage test.
| Kind | Candidate | 2-of-3, 10 | 2-of-3, 25 | 2-of-3, 50 | any-1-of-3, 10 | any, 25 | any, 50 |
|---|---|---|---|---|---|---|---|
| noise (false positives) | c=2, c=8, c=16, c=32, eager | 0% | 0% | 0% | 0% | 0% | 0% |
| noise (false positives) | chunk-256 | 0% | 0% | 0% | 0.6% | 0% | 0% |
| bad (detection) | rope 0.5x, FP8, FP8+KV, rope 0.1x | 100% | 100% | 100% | 100% | 100% | 100% |
*Finding:* on these data, 10 passages already separate every noise source from every bad change; the only false alarm seen in 6,000 noise subset draws per rule was 0.6% for one noise source (chunk-256) under the any-1-of-3 rule at 10 passages, and none under the 2-of-3 rule.
**Caveat (important):** the thresholds were chosen after looking at these same 50 passages and these same noise sources, so these rates are **in-sample**, not a forecast. A fair test needs new passages and new noise sources.

## What is the smallest damage the gate reliably catches?
Measured claim: **a change like rope_theta 0.5x (flip rate about 5%, mean KL about 0.008, perplexity about +0.6%) is caught on every subset of 10, 25 and 50 passages**, and the six noise sources are never failed (2-of-3 rule). Smaller damage than that was **not tested**: the thresholds imply the gate trips around flip rate 3.5%, KL 0.004 and perplexity +0.2%, but only when at least two of the metrics move that far together. The honest claim is "perplexity +0.6% or larger, on this model and text".

## Prediction vs measurement
| Prediction (written before measuring) | Predicted | Measured | Result |
|---|---|---|---|
| 1. FP8 and batch-shape bootstrap intervals overlap? | no overlap (no ranges guessed) | FP8 [7.35, 8.07] vs batch shape [1.56, 1.96] | correct |
| 2a. Largest of the four new noise sources | the c-sweep (c=2, 8, 16) | eager (2.06%); chunk-256 1.87%; c=16 1.53% | wrong |
| 2b. Any above c=32's 1.76%, or exactly zero? | "could go either way" | eager and chunk-256 above it; c=2 exactly zero | both happened |
| 3. New pod reference matches Day 11's exactly? | differs slightly (different physical GPU) | exactly zero | wrong |
| 4. Share of random 10-passage subsets where FP8 is caught (KL above 0.005) | 50 out of 100 | 100 out of 100 (every passage's FP8 KL is at least 0.012) | wrong |

## Self-check answers
1. **Why can't a bootstrap over passages tell you whether c=32 is a typical noise source?** It only resamples passages from the one c=32 capture, so it measures how the number changes with different text, not how it changes with a different noise source (c=2, eager, chunking). To learn about other noise sources I had to capture them: eager came out 17% larger than c=32.
2. **Why set the threshold above the largest noise measured rather than the average?** A line at the average would fail the noisier harmless sources (here, eager and chunk-256 are above the c=2/8/16 average), giving false alarms that teach people to ignore the gate. Since only six noise sources were measured, the real maximum may be larger still, so a margin is added.
3. **What happens to the false-positive rate if you test on only 10 passages instead of 50?** Each metric is averaged over fewer passages, so it wobbles more and noise can occasionally cross the line. Here the effect is tiny because the lines sit well above the noise: the only false alarm was 0.6% (chunk-256, any-1-of-3 rule, 10 passages); a line closer to the noise would show a larger effect.

## Code notes (Step 7)
**Not done.** I could not open vLLM's `check_logprobs_close` (in `tests/models/`) from my side, so the two questions (what it compares; how it decides "close enough" and how that differs from this gate) are open. To do: read the function and write a few sentences.

## What I did not do
- One model (0.6B), one text set (50 WikiText-2 passages), one GPU type; the top-20 KL is an approximation.
- No downstream-task evaluation: flips, KL and perplexity are proxies for quality.
- Only six noise sources and two rope settings; the thresholds were chosen on the same data used to score them (in-sample).
- Smaller damage than rope 0.5x was not tested.
- vLLM's own `check_logprobs_close` not read (above). Upstream PR hunt: **not started**.

## Files
`gate.py`, `test_gate.py`, `gate_thresholds.json`, `passage_metrics.py`, `bootstrap_day11.py` (+ `bootstrap_intervals.csv`, `subset_test.csv`), `candidate_table.py` / `.csv`, `scoreboard.py` / `scoreboard.csv` / `subset_rates.csv`, `captures/*.json.gz` (raw), `pod_info.txt`, `server_flags_check.txt`.
