# Day 9: Speculative decoding (n-gram, vLLM 0.32.0)

Model: Qwen3-0.6B (BF16) on one RTX 4090, `--max-model-len 4096`, prefix caching off, temperature 0, 200 output tokens with `ignore_eos`.
Script: `spec_bench.py <label> <copy|free> <concurrency>` (streams requests, measures TTFT/TPOT/throughput, and reads vLLM's speculative counters from `/metrics`).
Two workloads: **copy-heavy** (a ~300-token passage in the prompt, "repeat it exactly") and **free writing** ("write a story about ..."), at c=1 and c=32.
Every number is the average of two runs unless marked *single run*. Raw output: `results/raw_output.txt` and `results/*.json`. *Finding* = measured; *hypothesis* = not tested.

## Setup used
- Speculation: `--speculative-config '{"method":"ngram","num_speculative_tokens":K}'`. Read from `vllm/config/speculative.py`: the window `prompt_lookup_min/max` defaults to 5/5 when unset, so a guess needs the last 5 tokens to match earlier text.
- Baseline: the plain server. **Second baseline: `--no-async-scheduling`**, added after the server log said async scheduling is disabled whenever n-gram speculation is on (see "The confound").

## Results (c=1 and c=32, k=4)
| | Baseline TPOT (async on) | Baseline TPOT (async off) | k=4 TPOT | Speedup vs async-on | Speedup vs async-off | Acceptance rate | Tokens per step |
|---|---|---|---|---|---|---|---|
| Copy, c=1 | 2.14 ms | 3.61 ms | 0.87 ms | **2.47x** | **4.16x** | 0.88 | 4.53 |
| Free, c=1 | 2.14 ms | 3.53 ms *(single run)* | 1.67 ms | **1.28x** | 2.11x | 0.30 | 2.18 |
| Copy, c=32 | 3.99 ms | 6.60 ms *(single run)* | 1.80 ms | **2.22x** | 3.67x | 0.90 | 4.59 |
| Free, c=32 | 2.88 ms | 5.00 ms *(single run)* | 2.95 ms | **0.98x** | 1.70x | 0.28 | 2.14 |

Throughput speedups against the async-on baseline: copy c=1 2.32x, free c=1 1.27x, copy c=32 1.98x, free c=32 0.85x (about 15% lower).
TTFT at c=32 was lower with speculation (free: 41 ms vs 79 ms; copy: 82 vs 102 ms). The async-off baseline also had lower TTFT at c=32 (free 57 ms). I do not have an explanation.
Run-to-run: free c=32 throughput differed by about 10% between the two runs; the other cells agreed within about 2%.

## Prediction vs measurement
| Prediction (made before running) | Predicted | Measured | Gap |
|---|---|---|---|
| Copy: acceptance rate per guess (k=4) | 0.8 | 0.88 | +10% |
| Copy: tokens per verify step | 3.36 (formula, independent guesses) | 4.53 | +35% |
| Copy: verify step cost vs a normal step | 1.2x | about 1.8x (derived: 4.53 / 2.47) | +53% |
| Copy c=1 speedup | 2.8x (TPOT about 0.8 ms) | 2.47x (TPOT 0.87 ms) | -12% |
| Free c=1 speedup | 0.9 to 1.0x | 1.28x | about +35% |
| Copy c=32 speedup | 1.5 to 2x | 2.22x (TPOT), 1.98x (throughput) | above the range (about +27% vs the midpoint) |
| Identical outputs of 50 (speculation on vs off) | about 47 | 16 | -66% |

The copy c=1 prediction was close only because two errors cancelled: tokens per step were higher than predicted (+35%) and the step was far more expensive than predicted (+53%).
I made the step-cost estimate (1.2x) by stretching a line between the Day 8 c=1 and c=32 numbers; the Day 9 baseline later showed that line was too steep (0.024 ms per extra token, not 0.09).

## k sweep (copy-heavy, c=1; baseline 2.14 ms)
| k | TPOT | Speedup | Tokens per step | Acceptance per guess | Implied step time |
|---|---|---|---|---|---|
| 0 (baseline) | 2.14 ms | 1.0x | 1 | - | 2.14 ms |
| 2 | 1.34 ms | 1.59x | 2.85 | 0.93 | 3.83 ms |
| 4 | 0.87 ms | 2.47x | 4.53 | 0.88 | 3.93 ms |
| 8 | 0.53 ms | 4.04x | 7.46 | 0.81 | 3.96 ms |
| 16 | 0.34 ms | 6.29x | 12.07 | 0.69 | 4.10 ms |

(Implied step time = TPOT x tokens per step, derived.) *Finding:* **the gain had not leveled off by k=16.** The verify step stayed at 3.8 to 4.1 ms for every k, so extra guesses were almost free at c=1 while tokens per step kept growing.
Acceptance per guess declines as k grows, and the limit on this workload is how long the copy lasts, not the GPU.
Throughput gains are smaller than TPOT gains (k=16: 5.2x vs 6.3x) because the roughly 18 ms prefill (TTFT) is a fixed part of every request.
This is the best case (an exact copy). I swept k only on copy-heavy at c=1.

## The confound: async scheduling
The server log for every n-gram run says: "Async scheduling not supported with ngram-based speculative decoding and will be disabled." So the speculative server and the default baseline differ in more than speculation.
- *Finding:* with async scheduling off and **no speculation**, the plain baseline was 69% slower at c=1 (2.14 to 3.61 ms) and 65% slower at c=32 (3.99 to 6.60 ms); c=32 throughput fell 36% (7,037 to 4,505 tok/s).
- A plain step with async off (3.61 ms) is close to the speculative step (3.8 to 4.1 ms), so most of the "extra cost of speculation" I measured (about 1.7 ms) is the lost scheduling overlap: about 1.5 ms of it. Only about 0.3 ms remains for the lookup, verification and accept/reject.
- *Hypothesis:* async scheduling lets the CPU prepare the next step while the GPU runs the current one; without it the CPU time and GPU time add. This matches the CPU-launch costs seen on Days 4 to 7. I did not profile it.
- Which speedup to quote: the "vs async-on" column is what a user gets today by switching speculation on in this version; the "vs async-off" column is closer to the effect of speculation alone. Free writing at c=32 is a **loss against the default** (0.98x TPOT, 0.85x throughput) but a **win against the same-scheduler baseline** (1.70x).
- The async-off baselines for free c=1, copy c=32 and free c=32 are single runs.

## Lossless check (50 prompts, 100 tokens, temperature 0)
- Baseline vs baseline: **50 of 50 identical** (control).
- Speculation (k=4) vs baseline: **16 of 50 identical.** The 34 others first differ at token position 1 to 97 (median 23). That is roughly 1.2% of tokens flipping (my arithmetic on the positions, approximate); FP8 on Day 8 was about 5%.
- Speculation vs speculation: **50 of 50 identical**, so the difference is a fixed effect, not randomness.
- Reading differing pairs: both versions are weak outputs from a 0.6B model in raw-completion mode; "different, not clearly worse" (same as Day 8).
- *Hypothesis:* the accept/reject logic itself keeps only tokens equal to the target's own argmax (see code notes), so the differences must come from the target's logits changing slightly when 5 tokens go through at once instead of 1 (different kernels or shapes, so different rounding), flipping near-ties. Not tested.
  Another candidate I did not exclude: the speculative server also had async scheduling off while the quality baseline had it on. A baseline run with `--no-async-scheduling` would separate them. I predicted 47 of 50 because I assumed speculation barely changes the numbers; that was wrong by a wide margin.
- Quality was checked only at k=4.

## Self-check answers
1. *Why is verifying 5 tokens cheaper than generating 5 one at a time?* One step reads the model's weights from memory once and spends the idle arithmetic capacity on all 5 tokens. Five separate steps read the weights five times. (Day 8 numbers: 1 token per step took 2.27 ms, 32 tokens per step about 5.1 ms.)
2. *Why does n-gram do well on "repeat this paragraph" and badly on "write a story"?* The proposer only guesses what followed the same words earlier in the text. A repeated paragraph is already in the prompt, so guesses are right; a story is new text, so there is rarely a match.
   Measured: acceptance 0.88 on the copy, 0.30 on the story. My "badly" was too strong: 0.6B-model stories repeat themselves enough for 30% acceptance and a 1.28x gain at c=1.
3. *Why might speculation help less at c=32?* At c=32 the GPU's spare capacity is already partly used by 32 requests, so the extra guessed tokens cost real time. Measured: the gain at c=32 was smaller than at c=1 on free writing but about the same on copy-heavy (2.22x vs 2.47x). The scheduling confound muddies this comparison.

## Code notes
- **Proposer** (`vllm/v1/spec_decode/ngram_proposer.py`, `_find_longest_matched_ngram_and_propose_tokens`): it takes the tokens so far (prompt plus output) and looks for the longest run of the last `min..max` tokens that also appeared earlier in the text, using the earliest such position.
  If found, the guess is the k tokens that followed that earlier occurrence; if not (or if the text is shorter than the minimum), it returns nothing and the step is a normal one. It runs on the CPU (numba-compiled) and scans the whole text each step, which I infer adds CPU time that grows with context length.
- **Acceptance** (`vllm/v1/sample/rejection_sampler.py`, `rejection_greedy_sample_kernel`): for each guessed position it compares the guess to the target model's own top choice, writes the target's choice, and stops at the first mismatch. If all guesses match, it adds one bonus token from the target.
  Because every written token is the target's argmax, the method is lossless in the math at temperature 0.
- There is also an `ngram_proposer_gpu.py` and an `"ngram_gpu"` method name. I did not read or try it.

## What I did not do
- No draft-model stretch (Qwen3-4B target with Qwen3-0.6B draft).
- No profiler run, so the async-scheduling explanation and the 0.3 ms remainder are not decomposed.
- The k sweep covers copy-heavy at c=1 only; free writing and c=32 were run at k=4 only. The lookup window was left at its default of 5.
- `ngram_gpu` and a speculative run with async scheduling forced on were not tried.

## Files
`spec_bench.py`, `results/` (JSON per run, `raw_output.txt`), `outputs_base1.json`, `outputs_base2.json`, `outputs_spec1.json`, `outputs_spec2.json` (50 prompts each, from `../day-08-quantization/quality_gen.py`), `cmp_spec1.txt` (compare output, base1 vs spec1).
