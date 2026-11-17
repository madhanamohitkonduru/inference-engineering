"""No-GPU tests for gate.py:  python day-12-correctness-gate/test_gate.py"""
import gzip, json, os, subprocess, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate

V, K, NPASS, NPOS = 200, 20, 10, 60
rng = np.random.default_rng(0)
true_ids = rng.integers(0, V, size=(NPASS, NPOS))
base = rng.normal(0, 3, size=(NPASS, NPOS, V))


def fake_capture(path, logits):
    passages = []
    for p in range(NPASS):
        lp = logits[p] - np.log(np.exp(logits[p]).sum(axis=1, keepdims=True))
        order = np.argsort(-lp, axis=1)[:, :K]
        top = [[[int(j), float(lp[i, j])] for j in order[i]] for i in range(NPOS)]
        passages.append({"true_ids": true_ids[p].tolist(), "true_lp": [float(lp[i, true_ids[p, i]]) for i in range(NPOS)],
                         "top1": [t[0][0] for t in top], "top": top})
    with gzip.open(path, "wt") as fh:
        json.dump({"label": os.path.basename(path), "meta": {"n_passages": NPASS}, "passages": passages}, fh)


GOOD = {"flip_rate_pct": 4.0, "kl_mean": 0.005, "ppl_change_pct": 0.5, "min_metrics_failing": 2, "bootstrap_resamples": 200, "seed": 1}
with tempfile.TemporaryDirectory() as d:
    ref, same, tiny, big = [os.path.join(d, n + ".json.gz") for n in ("ref", "same", "tiny", "big")]
    fake_capture(ref, base); fake_capture(same, base)
    fake_capture(tiny, base + rng.normal(0, 0.005, base.shape)); fake_capture(big, base + rng.normal(0, 1.0, base.shape))
    th = os.path.join(d, "th.json"); json.dump(GOOD, open(th, "w"))
    assert gate.main([ref, same, "--thresholds", th]) == 0, "a capture against itself must pass"
    assert gate.main([ref, tiny, "--thresholds", th]) == 0, "tiny noise must pass"
    assert gate.main([ref, big, "--thresholds", th]) == 1, "large fake noise must fail"
    # units: the gate's numbers must equal the Day 11 compare tool's numbers (this caught a fraction-vs-percent bug)
    r_ = gate.evaluate(gate.pm_.lc.load_capture(ref), gate.pm_.lc.load_capture(big), GOOD)["metrics"]
    s_ = gate.pm_.lc.summarize(gate.pm_.lc.load_capture(ref), gate.pm_.lc.load_capture(big))
    assert abs(r_["flip_rate_pct"]["value"] - s_["flip_rate_pct"]) < 1e-9 and abs(r_["kl_mean"]["value"] - s_["approx_kl_mean"]) < 1e-12
    assert abs(r_["ppl_change_pct"]["value"] - 100 * s_["perplexity_diff"] / s_["perplexity_a"]) < 1e-9
    # rule check: with min_metrics_failing=3 a change that trips fewer than 3 metrics passes
    json.dump({**GOOD, "min_metrics_failing": 3, "ppl_change_pct": 1e9}, open(th, "w"))
    assert gate.main([ref, big, "--thresholds", th]) == 0, "needs all 3 to fail; perplexity threshold is huge so only 2 fail -> pass"
    # missing key: clear error, exit code 2
    bad = {k: v for k, v in GOOD.items() if k != "kl_mean"}; json.dump(bad, open(th, "w"))
    r = subprocess.run([sys.executable, os.path.join(HERE, "gate.py"), ref, same, "--thresholds", th], capture_output=True, text=True)
    assert r.returncode == 2 and "missing required key(s): kl_mean" in r.stderr, (r.returncode, r.stderr)
    r = subprocess.run([sys.executable, os.path.join(HERE, "gate.py"), "nope", same, "--thresholds", th], capture_output=True, text=True)
    assert r.returncode == 2
print("all gate tests passed")
