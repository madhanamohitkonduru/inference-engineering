"""Per-passage metrics and bootstrap helpers (reused by bootstrap_day11.py and, later, gate.py).
Builds on day-11-correctness-harness/logprob_compare.py (same capture format)."""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "day-11-correctness-harness"))
import logprob_compare as lc


def per_passage(ref, cand):
    """Compare two loaded captures; return arrays with one row per passage (columns = positions)."""
    n = ref["meta"]["n_passages"]
    pp = lc.per_position(ref, cand)                       # per position: abs_diff, flip, kl
    shape = (n, -1)
    return {"flip": pp["flip"].reshape(shape).astype(float), "kl": pp["kl"].reshape(shape),
            "abs_diff": pp["abs_diff"].reshape(shape),
            "lp_ref": ref["true_lp"].reshape(shape), "lp_cand": cand["true_lp"].reshape(shape)}


def passage_means(pm):
    """One number per passage for each metric (every passage has the same number of positions)."""
    return {"flip_rate": pm["flip"].mean(1), "kl": pm["kl"].mean(1), "abs_diff": pm["abs_diff"].mean(1),
            "lp_ref": pm["lp_ref"].mean(1), "lp_cand": pm["lp_cand"].mean(1)}


def overall(pmeans, idx):
    """Overall metrics for a selection of passages. idx: (n_sets, k) passage indices (repeats allowed)."""
    return {"flip_rate": pmeans["flip_rate"][idx].mean(1), "kl": pmeans["kl"][idx].mean(1),
            "abs_diff": pmeans["abs_diff"][idx].mean(1),
            "ppl_change": np.exp(-pmeans["lp_cand"][idx].mean(1)) - np.exp(-pmeans["lp_ref"][idx].mean(1))}


def bootstrap_idx(n, k, n_sets, rng):
    return rng.integers(0, n, size=(n_sets, k))                 # with replacement


def subset_idx(n, k, n_sets, rng):
    return np.argsort(rng.random((n_sets, n)), axis=1)[:, :k]   # without replacement
