#!/usr/bin/env python3
"""Step 5 analysis: pair table, histogram, most-changed positions, free-run prediction. Run from the repo root."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logprob_compare as lc

D = os.path.dirname(os.path.abspath(__file__)); CAP = os.path.join(D, "captures")
REF = "bf16_c1_run1"
PAIRS = [("control (BF16 run 2)", "bf16_c1_run2"), ("batch shape (BF16 c=32)", "bf16_c32"),
         ("FP8 weights", "fp8_c1"), ("FP8 weights + FP8 KV", "fp8kv_c1")]
cap = {n: lc.load_capture(os.path.join(CAP, n + ".json.gz")) for n in [REF] + [p[1] for p in PAIRS]}
pp = {n: lc.per_position(cap[REF], cap[n]) for _, n in PAIRS}
POS_PER_PASSAGE = 511

# ---- 1. table
rows = []
for name, n in PAIRS:
    s = lc.summarize(cap[REF], cap[n]); rows.append((name, s))
with open(os.path.join(D, "pair_table.csv"), "w") as f:
    f.write("pair,positions,mean_abs_logprob_diff,max_abs_logprob_diff,flip_rate_pct,approx_kl_mean,perplexity_ref,perplexity_pair,perplexity_diff\n")
    for name, s in rows:
        f.write(f"{name},{s['positions']},{s['mean_abs_logprob_diff']:.6f},{s['max_abs_logprob_diff']:.4f},{s['flip_rate_pct']:.3f},"
                f"{s['approx_kl_mean']:.8f},{s['perplexity_a']:.4f},{s['perplexity_b']:.4f},{s['perplexity_diff']:+.4f}\n")
        print(f"{name:26s} diff {s['mean_abs_logprob_diff']:.4f} (max {s['max_abs_logprob_diff']:.2f}) flip {s['flip_rate_pct']:.2f}% KL {s['approx_kl_mean']:.5f} ppl {s['perplexity_diff']:+.4f}")

# ---- 2. histogram (log y)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
top = max(pp[n]["abs_diff"].max() for _, n in PAIRS[1:])
bins = np.linspace(0, top, 60)
for a, (name, n) in zip(ax, PAIRS[1:]):
    d = pp[n]["abs_diff"]
    a.hist(d, bins=bins); a.set_yscale("log"); a.set_title(name, fontsize=10)
    a.set_xlabel("|true-token logprob difference| vs BF16 c=1")
    q = np.quantile(d, [.5, .99, .999])
    a.text(0.97, 0.95, f"p50 {q[0]:.3f}\np99 {q[1]:.2f}\np99.9 {q[2]:.2f}\nmax {d.max():.2f}", transform=a.transAxes, ha="right", va="top", fontsize=8)
ax[0].set_ylabel("positions (log scale)")
fig.suptitle("Per-token logprob change vs BF16 c=1 (25,550 positions; control is all zeros)")
fig.tight_layout(); fig.savefig(os.path.join(D, "logprob_diff_hist.png"), dpi=150)
for name, n in PAIRS[1:]:
    d = pp[n]["abs_diff"]
    print(f"{name:26s} share >0.1: {100*(d>0.1).mean():.2f}%  >0.5: {100*(d>0.5).mean():.2f}%  >1: {100*(d>1).mean():.2f}%  p50 {np.median(d):.4f} p99 {np.quantile(d,.99):.3f} p99.9 {np.quantile(d,.999):.3f} max {d.max():.2f}")

# ---- 3. ten most-changed positions under FP8
P = json.load(open(os.path.join(D, "passages.json")))["passages"]
vocab = {i: p for pas in P for i, p in zip(pas["ids"], pas["pieces"])}
tokname = lambda i: repr(vocab.get(int(i), f"#{int(i)}"))
a, b = cap[REF], cap["fp8_c1"]
order = np.argsort(-pp["fp8_c1"]["abs_diff"])[:10]
lines = []
for k, g in enumerate(order, 1):
    pi, pos = divmod(int(g), POS_PER_PASSAGE)
    ids, pieces = P[pi]["ids"], P[pi]["pieces"]
    ctx = "".join(pieces[max(0, pos - 11):pos + 1]).replace("\n", "\\n")
    tid = int(a["true_ids"][g])
    def top3(c):
        t = c["top"][g][:3]; return ", ".join(f"{tokname(i)} {100*np.exp(lp):.1f}%" for i, lp in t)
    ta, tb = a["top"][g], b["top"][g]
    lines.append(f"#{k} passage {pi} position {pos+1}: ...{ctx}[{tokname(tid)}]\n"
                 f"   true token logprob: BF16 {a['true_lp'][g]:.2f} ({100*np.exp(a['true_lp'][g]):.1f}%)  FP8 {b['true_lp'][g]:.2f} ({100*np.exp(b['true_lp'][g]):.1f}%)\n"
                 f"   BF16 top3: {top3(a)}\n   FP8  top3: {top3(b)}\n"
                 f"   top1-top2 gap (prob points): BF16 {100*(np.exp(ta[0][1])-np.exp(ta[1][1])):.1f}  FP8 {100*(np.exp(tb[0][1])-np.exp(tb[1][1])):.1f}   flip: {bool(pp['fp8_c1']['flip'][g])}")
open(os.path.join(D, "top10_changed_fp8.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# ---- 3b. are flips near-ties? top1-top2 probability gap (BF16 side) at flipped vs all positions
gap = lambda c: np.exp(c["top"][:, 0, 1]) - np.exp(c["top"][:, 1, 1])
ga = gap(a)
for name, n in PAIRS[1:]:
    fl = pp[n]["flip"]
    print(f"{name:26s} BF16 top1-top2 gap: at flips median {100*np.median(ga[fl]):.1f} pts, share of flips with gap<5pts {100*(ga[fl]<0.05).mean():.0f}%, "
          f"all positions median {100*np.median(ga):.1f} pts, share <5pts {100*(ga<0.05).mean():.0f}%")

# ---- 4. free-run prediction
print("\nfree-run prediction, 100-token outputs, 50 outputs")
fr = {}
for name, n in PAIRS[1:]:
    p = pp[n]["flip"].mean()
    fl = pp[n]["flip"].reshape(-1, POS_PER_PASSAGE)[:, :100]
    emp = int((fl.sum(axis=1) == 0).sum())
    fr[name] = (p, (1 - p) ** 100 * 50, emp)
    print(f"{name:26s} flip {100*p:.2f}%  formula (1-p)^100 x50 = {(1-p)**100*50:.2f}   passages with no flip in first 100 positions: {emp}/50")
json.dump({k: {"flip_rate": v[0], "expected_identical_of_50": v[1], "empirical_no_flip_first100_of_50": v[2]} for k, v in fr.items()},
          open(os.path.join(D, "freerun_prediction.json"), "w"), indent=1)
