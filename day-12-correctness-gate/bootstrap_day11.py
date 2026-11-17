#!/usr/bin/env python3
"""Step 3 (no GPU): per-passage metrics, 95% bootstrap intervals, and the small-subset false-alarm / miss test on Day 11's captures.
Run from anywhere:  python day-12-correctness-gate/bootstrap_day11.py"""
import os
import numpy as np
import passage_metrics as pm_

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.join(HERE, "..", "day-11-correctness-harness", "captures")
REF = "bf16_c1_run1"
PAIRS = [("batch shape (BF16 c=32)", "bf16_c32"), ("FP8 weights", "fp8_c1"), ("FP8 weights + FP8 KV", "fp8kv_c1")]
KL_THRESHOLD = 0.005          # Day 11 candidate threshold
B = 1000
rng = np.random.default_rng(12)

ref = pm_.lc.load_capture(os.path.join(CAP, REF + ".json.gz"))
means = {}
for name, label in PAIRS:
    cand = pm_.lc.load_capture(os.path.join(CAP, label + ".json.gz"))
    means[name] = pm_.passage_means(pm_.per_passage(ref, cand))
n = len(next(iter(means.values()))["kl"])

# ---- per-passage spread (why subsets of 10 can mislead)
print(f"{n} passages. Per-passage mean KL (min / median / max) and passages with KL below {KL_THRESHOLD}:")
for name in means:
    k = means[name]["kl"]
    print(f"  {name:26s} {k.min():.5f} / {np.median(k):.5f} / {k.max():.5f}   below threshold: {(k < KL_THRESHOLD).sum()}/{n}")
print("Per-passage flip rate % (min / median / max):")
for name in means:
    f = 100 * means[name]["flip_rate"]; print(f"  {name:26s} {f.min():.1f} / {np.median(f):.1f} / {f.max():.1f}")

# ---- bootstrap intervals: resample the 50 passages with replacement, 1,000 times
idx = pm_.bootstrap_idx(n, n, B, rng)
rows = ["pair,metric,point,ci_low_2.5,ci_high_97.5"]
print("\n95% bootstrap intervals (1,000 resamples of the 50 passages)")
for name in means:
    point = pm_.overall(means[name], np.arange(n)[None, :])
    boot = pm_.overall(means[name], idx)
    for m, scale, fmt in [("flip_rate", 100, "{:.2f}"), ("kl", 1, "{:.5f}"), ("ppl_change", 1, "{:+.3f}")]:
        p, lo, hi = point[m][0] * scale, *np.percentile(boot[m] * scale, [2.5, 97.5])
        rows.append(f"{name},{m},{fmt.format(p)},{fmt.format(lo)},{fmt.format(hi)}")
        print(f"  {name:26s} {m:11s} {fmt.format(p)}   [{fmt.format(lo)}, {fmt.format(hi)}]")
open(os.path.join(HERE, "bootstrap_intervals.csv"), "w").write("\n".join(rows) + "\n")

# ---- small-subset test at the Day 11 KL threshold
print(f"\nSubset test, mean-KL threshold {KL_THRESHOLD} ({B} random subsets each). false alarm = batch shape above threshold; miss = bad change at or below it")
rows = ["subset_size,sampling,batch_shape_false_alarm_pct,fp8_miss_pct,fp8kv_miss_pct"]
for k in (10, 25, 50):
    if k < n:
        ids, how = pm_.subset_idx(n, k, B, rng), "without replacement"
    else:
        ids, how = pm_.bootstrap_idx(n, k, B, rng), "with replacement (bootstrap)"
    res = {nm: pm_.overall(means[nm], ids)["kl"] for nm in means}
    fa = 100 * (res["batch shape (BF16 c=32)"] > KL_THRESHOLD).mean()
    m1 = 100 * (res["FP8 weights"] <= KL_THRESHOLD).mean()
    m2 = 100 * (res["FP8 weights + FP8 KV"] <= KL_THRESHOLD).mean()
    rows.append(f"{k},{how},{fa:.1f},{m1:.1f},{m2:.1f}")
    print(f"  k={k:2d} ({how:28s}) batch-shape false alarms {fa:5.1f}%   FP8 missed {m1:5.1f}%   FP8+KV missed {m2:5.1f}%")
open(os.path.join(HERE, "subset_test.csv"), "w").write("\n".join(rows) + "\n")
