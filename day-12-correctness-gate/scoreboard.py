#!/usr/bin/env python3
"""Step 6b/6c: score every candidate with gate_thresholds.json (point-value scoreboard, then false-positive / detection rates on random passage subsets).
Run:  python day-12-correctness-gate/scoreboard.py"""
import os, json
import numpy as np
import passage_metrics as pm_
import gate

HERE = os.path.dirname(os.path.abspath(__file__))
D12, D11 = os.path.join(HERE, "captures"), os.path.join(HERE, "..", "day-11-correctness-harness", "captures")
REF = os.path.join(D12, "new_bf16_c1.json.gz")
CANDS = [("noise", "c=2", D12, "noise_c2"), ("noise", "c=8", D12, "noise_c8"), ("noise", "c=16", D12, "noise_c16"),
         ("noise", "c=32", D11, "bf16_c32"), ("noise", "chunk-256", D12, "noise_chunk256_c1"), ("noise", "eager", D12, "noise_eager_c1"),
         ("bad", "rope 0.5x (mild)", D12, "bad_rope_mild_c1"), ("bad", "FP8", D11, "fp8_c1"),
         ("bad", "FP8 + KV", D11, "fp8kv_c1"), ("bad", "rope 0.1x (big)", D12, "bad_rope_big_c1")]
th = json.load(open(os.path.join(HERE, "gate_thresholds.json")))
KEYS = [("flip_rate_pct", "flip_rate", 100.0), ("kl_mean", "kl", 1.0), ("ppl_change_pct", "ppl_change_pct", 1.0)]
ref = pm_.lc.load_capture(REF)
means = {n: pm_.passage_means(pm_.per_passage(ref, pm_.lc.load_capture(os.path.join(d, l + ".json.gz")))) for _, n, d, l in CANDS}
n = len(next(iter(means.values()))["kl"])

# ---- point-value scoreboard
rows = ["kind,candidate,flip_fail,kl_fail,ppl_fail,metrics_failed,verdict_2of3,verdict_any"]
print(f"thresholds: flip {th['flip_rate_pct']}%  KL {th['kl_mean']}  ppl +{th['ppl_change_pct']}%   rule: fail if >= {th['min_metrics_failing']} of 3\n")
print(f"{'kind':6s}{'candidate':20s}{'flip':>8s}{'KL':>10s}{'ppl%':>8s}   fails  2-of-3  any")
for kind, name, _, _ in CANDS:
    o = pm_.overall(means[name], np.arange(n)[None, :])
    fails = [bool(o[c][0] * s > th[k]) for k, c, s in KEYS]
    f = sum(fails); v2, v1 = ("FAIL" if f >= 2 else "PASS"), ("FAIL" if f >= 1 else "PASS")
    rows.append(f"{kind},{name},{fails[0]},{fails[1]},{fails[2]},{f},{v2},{v1}")
    print(f"{kind:6s}{name:20s}{o['flip_rate'][0]*100:8.2f}{o['kl'][0]:10.5f}{o['ppl_change_pct'][0]:8.3f}   {f}/3    {v2}    {v1}")
open(os.path.join(HERE, "scoreboard.csv"), "w").write("\n".join(rows) + "\n")

# ---- subsets: how often does the gate FAIL a candidate, by number of passages tested
rng = np.random.default_rng(2012); B = 1000
rows = ["kind,candidate,rule,subset_size,gate_fails_pct"]
print("\n% of random passage subsets on which the gate FAILS the candidate (noise rows = false-positive rate, bad rows = detection rate)")
print(f"{'kind':6s}{'candidate':20s}" + "".join(f"{r+' k='+str(k):>14s}" for r in ("2of3", "any") for k in (10, 25, 50)))
idxs = {k: (pm_.subset_idx(n, k, B, rng) if k < n else pm_.bootstrap_idx(n, k, B, rng)) for k in (10, 25, 50)}
for kind, name, _, _ in CANDS:
    cells = []
    for rule, need in (("2of3", 2), ("any", 1)):
        for k in (10, 25, 50):
            o = pm_.overall(means[name], idxs[k])
            nf = sum((o[c] * s > th[key]).astype(int) for key, c, s in KEYS)
            pct = 100 * (nf >= need).mean(); cells.append(pct)
            rows.append(f"{kind},{name},{rule},{k},{pct:.1f}")
    print(f"{kind:6s}{name:20s}" + "".join(f"{c:13.1f}%" for c in cells))
open(os.path.join(HERE, "subset_rates.csv"), "w").write("\n".join(rows) + "\n")
