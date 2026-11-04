"""Tests for the compare half, no GPU needed:  python day-11-correctness-harness/test_logprob_compare.py"""
import gzip, json, os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logprob_compare as lc

V, K, NPASS, NPOS = 200, 20, 3, 40
rng = np.random.default_rng(0)
true_ids = rng.integers(0, V, size=(NPASS, NPOS))
base_logits = rng.normal(0, 3, size=(NPASS, NPOS, V))


def fake_capture(label, logits, outdir):
    passages = []
    for p in range(NPASS):
        lp_all = logits[p] - np.log(np.exp(logits[p]).sum(axis=1, keepdims=True))      # log-softmax
        true_lp = [float(lp_all[i, true_ids[p, i]]) for i in range(NPOS)]
        order = np.argsort(-lp_all, axis=1)[:, :K]
        top = [[[int(j), float(lp_all[i, j])] for j in order[i]] for i in range(NPOS)]
        passages.append({"true_ids": true_ids[p].tolist(), "true_lp": true_lp, "top1": [t[0][0] for t in top], "top": top})
    with gzip.open(os.path.join(outdir, label + ".json.gz"), "wt") as fh:
        json.dump({"label": label, "meta": {}, "passages": passages}, fh)


with tempfile.TemporaryDirectory() as d:
    fake_capture("base", base_logits, d)
    fake_capture("same", base_logits, d)
    fake_capture("tiny", base_logits + rng.normal(0, 0.01, base_logits.shape), d)
    fake_capture("big", base_logits + rng.normal(0, 1.0, base_logits.shape), d)
    load = lambda n: lc.load_capture(os.path.join(d, n + ".json.gz"))
    s0 = lc.summarize(load("base"), load("base"))           # a file against itself must be exactly zero
    assert s0["mean_abs_logprob_diff"] == 0 and s0["flip_rate_pct"] == 0 and s0["approx_kl_mean"] == 0 and s0["perplexity_diff"] == 0, s0
    s1 = lc.summarize(load("base"), load("same"))
    assert s1["mean_abs_logprob_diff"] == 0 and s1["flip_rate_pct"] == 0 and s1["approx_kl_mean"] == 0
    st, sb = lc.summarize(load("base"), load("tiny")), lc.summarize(load("base"), load("big"))
    assert 0 < st["mean_abs_logprob_diff"] < sb["mean_abs_logprob_diff"]
    assert 0 < st["approx_kl_mean"] < sb["approx_kl_mean"]
    assert st["flip_rate_pct"] <= sb["flip_rate_pct"]
    print("self vs self:", s0)
    print("tiny noise  :", {k: round(v, 6) for k, v in st.items()})
    print("big noise   :", {k: round(v, 6) for k, v in sb.items()})
print("all tests passed")
