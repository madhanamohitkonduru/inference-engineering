#!/usr/bin/env python3
"""Step 6a: one row per candidate vs the new reference: flip rate, mean KL, perplexity change, with 95% bootstrap intervals (passages resampled).
Run:  python day-12-correctness-gate/candidate_table.py"""
import os, json
import numpy as np
import passage_metrics as pm_
import gate

HERE = os.path.dirname(os.path.abspath(__file__))
D12, D11 = os.path.join(HERE, "captures"), os.path.join(HERE, "..", "day-11-correctness-harness", "captures")
REF = os.path.join(D12, "new_bf16_c1.json.gz")
ROWS = [("noise", "c=2", D12, "noise_c2"), ("noise", "c=8", D12, "noise_c8"), ("noise", "c=16", D12, "noise_c16"),
        ("noise", "c=32 (Day 11 capture)", D11, "bf16_c32"), ("noise", "chunk-256", D12, "noise_chunk256_c1"), ("noise", "eager", D12, "noise_eager_c1"),
        ("bad", "rope 0.5x (mild)", D12, "bad_rope_mild_c1"), ("bad", "FP8 weights (Day 11 capture)", D11, "fp8_c1"),
        ("bad", "FP8 + FP8 KV (Day 11 capture)", D11, "fp8kv_c1"), ("bad", "rope 0.1x (big)", D12, "bad_rope_big_c1")]
th = json.load(open(os.path.join(HERE, "gate_thresholds.json")))
ref = pm_.lc.load_capture(REF)
out = ["kind,candidate,flip_rate_pct,flip_lo,flip_hi,kl_mean,kl_lo,kl_hi,ppl_change_pct,ppl_lo,ppl_hi"]
print(f"{'kind':6s}{'candidate':32s}{'flip % [95% CI]':26s}{'mean KL [95% CI]':32s}{'perplexity change % [95% CI]'}")
for kind, name, d, label in ROWS:
    r = gate.evaluate(ref, pm_.lc.load_capture(os.path.join(d, label + ".json.gz")), th)["metrics"]
    f, k, p = r["flip_rate_pct"], r["kl_mean"], r["ppl_change_pct"]
    out.append(f"{kind},{name},{f['value']:.3f},{f['ci_low']:.3f},{f['ci_high']:.3f},{k['value']:.6f},{k['ci_low']:.6f},{k['ci_high']:.6f},{p['value']:.3f},{p['ci_low']:.3f},{p['ci_high']:.3f}")
    print(f"{kind:6s}{name:32s}{f['value']:6.2f} [{f['ci_low']:5.2f}, {f['ci_high']:5.2f}]".ljust(64) + f"{k['value']:.5f} [{k['ci_low']:.5f}, {k['ci_high']:.5f}]".ljust(32) + f"{p['value']:+.3f} [{p['ci_low']:+.3f}, {p['ci_high']:+.3f}]")
open(os.path.join(HERE, "candidate_table.csv"), "w").write("\n".join(out) + "\n")
