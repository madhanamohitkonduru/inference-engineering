"""Day 4: where does one decode step go at c=1? Builds the per-step time budget from the PyTorch-profiler totals
(see pytorch-profiler/summary_c1.md) and compares each part with the memory-bandwidth ideal.  python3 decompose.py
"""
BW = 1008e9
# Data each part must read per decode step (bytes). Shapes from Qwen3-0.6B config.json.
layer_bytes = 28 * (1024*4096 + 2048*1024 + 1024*6144 + 3072*1024) * 2      # 880.8 MB
lm_head_bytes = 151_936 * 1024 * 2                                         # 311.2 MB
kv_bytes = 576 * 114_688                                                    # 66.1 MB (avg context 576)

# Profiler totals (c=1 trace, 293 decode steps + 3 prompt steps)
decode_steps = 293
gemvx_total_ms, lm_head_total_ms, lm_head_calls = 436.095, 97.007, 296     # gemvx includes the lm_head matmul
attn_ms = 59.296 + 55.355                                                   # splitkv + combine
kernel_total_ms, prompt_steps_ms = 674.5, 3 * 4.751                         # summed kernel time; 3 prompt iterations
idle_per_step_ms = 0.099                                                    # Nsight, graphs on, c=1

decode_kernel_ms = (kernel_total_ms - prompt_steps_ms) / decode_steps
layer_ms = (gemvx_total_ms - lm_head_total_ms) / decode_steps
lm_ms = lm_head_total_ms / lm_head_calls
att_ms = attn_ms / decode_steps
other_ms = decode_kernel_ms - layer_ms - lm_ms - att_ms

rows = [("Layer matmuls (112/step)", layer_ms, layer_bytes),
        ("lm_head matmul", lm_ms, lm_head_bytes),
        ("Attention (56 kernels/step)", att_ms, kv_bytes),
        ("Other small kernels", other_ms, 0),
        ("Idle gaps (Nsight)", idle_per_step_ms, 0)]
ideal = {n: b / BW * 1e3 for n, _, b in rows}
total, ideal_total = sum(r[1] for r in rows), sum(ideal.values())
extra = total - ideal_total

print(f"| Part | Time/step (ms) | Ideal at 1,008 GB/s (ms) | Extra (ms) | Share of extra | Achieved GB/s |")
print("|---|---|---|---|---|---|")
for n, t, b in rows:
    e = t - ideal[n]
    ach = f"{b/(t/1e3)/1e9:,.0f} ({100*b/(t/1e3)/BW:.0f}%)" if b else "n/a"
    print(f"| {n} | {t:.3f} | {ideal[n]:.3f} | {e:+.3f} | {100*e/extra:.0f}% | {ach} |")
print(f"| **Total** | **{total:.3f}** | **{ideal_total:.3f}** | **{extra:+.3f}** | 100% | |")
print(f"\nMeasured TPOT without profiler: 2.26 ms; profile-based total {total:.2f} ms ({(total/2.26-1)*100:+.0f}%).")
