# Day 8: Quantization (FP8 weights, FP8 KV cache)

Model: Qwen3-0.6B (BF16 checkpoint) on one RTX 4090, vLLM 0.32.0, `--max-model-len 4096`, prefix caching off.
Benchmark: 100 prompts, random 512 tokens in / 128 out, at concurrency 1 and 32 (`scripts/bench_quick.sh`).
Every number is the average of two runs of the same server (A, B, C each run twice). Raw output: `results/raw_output.txt`.
*Finding* = measured. *Hypothesis* = not yet tested.

## Configs
| | Flags |
|---|---|
| A | BF16 (baseline) |
| B | `--quantization fp8` |
| C | `--quantization fp8 --kv-cache-dtype fp8` |
| D | 4-bit: **not measured** (see "What I did not do") |

## Results and prediction vs measurement
| | A | B | C |
|---|---|---|---|
| Weights in GPU memory | 1.12 GiB | 0.71 GiB | 0.71 GiB |
| KV pool (tokens) | 167,728 | 170,256 | 333,936 |
| c=1 TPOT | 2.27 ms | 2.06 ms (-9.3%) | 1.96 ms (-13.7%) |
| c=1 throughput | 400.6 tok/s | 429.4 (+7.2%) | 459.8 (+14.8%) |
| c=1 TTFT | 31.3 ms | 35.6 ms (+13.7%) | 28.7 ms (-8.2%) |
| c=32 TPOT | 5.10 ms | 5.43 ms (+6.6%) | 4.07 ms (-20.2%) |
| c=32 throughput | 4,416 tok/s | 4,243 (-3.9%) | 5,253 (+19.0%) |
| c=32 TTFT | 164.4 ms | 171.8 ms (+4.5%) | 168.9 ms (+2.7%) |

c=32 TTFT differences are within the run-to-run spread (two runs differ by up to 11 ms), so I do not read anything into them.

| Prediction | Predicted | Measured | Gap |
|---|---|---|---|
| B weight memory | 0.71 GiB | 0.71 GiB | 0% |
| B KV pool | ~171,600 | 170,256 | -0.8% |
| C KV pool | ~343,100 | 333,936 | -2.7% |
| B c=1 TPOT (bytes model: -18.6%) | 1.85 ms | 2.06 ms | measured speedup was half the prediction (-9.3% vs -18.6%) |
| Quality: identical outputs of 50 (B, my guess) | ~30 | 3 | far off |
| 4-bit (D) c=1 TPOT ceiling | -27.8% (~1.64 ms) | not measured | - |

Run-to-run noise in the KV pool: the same config gave 167,728 and 169,712 (A, +1.2%) and 170,256 and 172,016 (B, +1.0%) on two launches.
So B's pool (+1.5% over A) is mostly noise; C's 1.99x is far outside it.

## Findings (measured)
- Halving the layer weights saved exactly the predicted memory (1.12 to 0.71 GiB). `lm_head`/embedding stay BF16, which is why the total does not halve.
- FP8 weights made c=1 decode faster, but by **half** the bytes-model prediction (-9.3% vs -18.6%), and made c=32 **slower** (+6.6% TPOT).
- Adding the FP8 KV cache doubled the KV pool (1.99x) and gave the largest win at c=32 (TPOT 5.43 to 4.07 ms against B, 5.10 to 4.07 against A).
- Quality: two BF16 runs were identical (50/50), so every difference below comes from quantization.
  B vs A: 3/50 identical, C vs A: 1/50, C vs B: 2/50. First divergence is early (median token 8 to 9).

## Hypotheses (not tested)
- **Why FP8 weights help less than the bytes model says.** Each of the 112 quantized matrices needs its input converted to FP8 every step (dynamic activation scale),
  and FP8 kernels at tiny sizes may be less efficient. Test: profile with the Day 4 tools and count the extra kernels.
- **Why FP8 KV helps at c=32 and barely at c=1.** Back-of-envelope: at c=32 with ~576 tokens of context, one step reads about 2.1 GB of KV at BF16
  (576 x 32 x 114,688 B); halving that is ~1.05 GB, about 1.05 ms at 1,008 GB/s. The measured saving, 5.09 to 4.07 ms (1.0 ms), is close. At c=1 the KV read is only 66 MB.
- **Why outputs diverge so much.** Raw-completion text from a 0.6B model has many near-tied next tokens, so any small numeric change flips some of them,
  and the text then goes down a different path. About 4 to 6% of tokens flip (rough, from token positions found by re-tokenizing). This would predict that other small changes
  (a different kernel or batch size on BF16) also change outputs. Not tested.
- "Identical output" measures similarity, not correctness. Reading the pairs, differences were mostly "different, not clearly worse" (the BF16 outputs are weak too).
  Proper quality measurement is Day 11.

## Code notes: which FP8 path actually ran
- `--quantization fp8` printed a deprecation warning and was redirected to `fp8_per_tensor`. The class that ran is `Fp8PerTensorOnlineLinearMethod`
  (`quantization/online/fp8.py`), with `CutlassFP8ScaledMMLinearKernel`. I first read `Fp8Config`/`Fp8LinearMethod` in `quantization/fp8.py` by mistake:
  that is the path for checkpoints already saved in FP8, and `Fp8Config` raises an error for online quantization. Lesson: check the server log to see which code ran.
- Which layers: the log says "Quantizing 112 layers" = 28 layers x 4 matrices (`mlp.down_proj`, `mlp.gate_up_proj` and, by the cut-off line, the two attention matrices).
  Embedding and `lm_head` are not listed and stay BF16. The memory arithmetic agrees: 440 MB + 311 MB = 0.70 GiB.
- Scale: in `process_weights_after_loading` the code takes the largest absolute value of the whole matrix (`weight_amax`), divides by the FP8 maximum (`_fp8_scale`),
  and calls `ops.scaled_fp8_quant(weight, scale=weight_scale)`. One scale per matrix, computed once at load (the log says 0.07 s).
  A fused matrix (`gate_up_proj`) therefore shares one scale. I assumed the FP8 maximum is 448 and did not read `_fp8_max`.
- Activations: `layer.input_scale = None` and `apply()` just calls `fp8_linear.apply_weights`. I infer the activation scale is computed per step by the kernel (dynamic), but did not read the kernel.

## Self-check notes
- Decode is memory-bound, so fewer bytes per step should help; the final `lm_head` layer is a large share of those bytes and is not quantized here.
- My bytes-model prediction for B (-18.6%) assumed the non-weight overhead stays the same. It was too optimistic; I recorded that rather than adjusting it afterwards.

## What I did not do
- **Config D (4-bit) was skipped.** The only Qwen3-0.6B GPTQ checkpoints I found (`Efficient-ML/Qwen3-0.6B-gptq-w4-128` and siblings) contain a research `.pth` file and no `config.json`, so vLLM refused to load them.
  The -27.8% figure above is a ceiling from the bytes model, not a result, and FP8 already reached only half of its predicted gain.
- No profiler run on FP8 (the hypotheses above are untested).
- C and B have two runs each; differences under about 2% should not be trusted.

## What to expect on a larger model (hypothesis)
In a larger model each matrix holds far more weights, while the per-layer work of converting the input grows much more slowly, so FP8 weights should help more than here and the fixed per-layer extra work should matter relatively less. A larger model also leaves less room for the KV pool, so FP8 KV should matter more. I have not measured any of this.

## Files
- `quality_gen.py`, `quality_compare.py`: generate and compare 50 fixed prompts at temperature 0.
- `outputs_A_run1.json`, `outputs_A_run2.json`, `outputs_B.json`, `outputs_C.json`: raw generated text (copied from the pod).
- `results/raw_output.txt`: benchmark and server-log lines.
