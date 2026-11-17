#!/usr/bin/env python3
"""Correctness gate: compare a candidate capture against a reference capture and return PASS or FAIL.

  python day-12-correctness-gate/gate.py REFERENCE CANDIDATE --thresholds day-12-correctness-gate/gate_thresholds.json

REFERENCE / CANDIDATE: a capture path (.json.gz) or a label such as bf16_c1_run1 (searched in --captures-dir, default: the Day 11 and Day 12 capture folders).
Metrics (candidate vs reference, teacher-forced logprobs):  flip_rate_pct, kl_mean (approximate, top-20), ppl_change_pct (one-sided: only a rise counts).
Decision: a metric FAILS when its point value is above its threshold; the gate FAILS when at least `min_metrics_failing` metrics fail.
The 95% bootstrap interval (resampling passages) is printed for information; the decision uses the point value.
Exit code: 0 = PASS, 1 = FAIL, 2 = error (bad input or thresholds file), so CI can tell a failed gate from a broken one."""
import argparse, json, os, sys
import numpy as np
import passage_metrics as pm_

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIRS = [os.path.join(HERE, "captures"), os.path.join(HERE, "..", "day-11-correctness-harness", "captures")]
METRICS = [("flip_rate_pct", "flip_rate", "flip rate (%)", 100.0), ("kl_mean", "kl", "mean KL", 1.0), ("ppl_change_pct", "ppl_change_pct", "perplexity change (%)", 1.0)]  # last = scale to the unit of the threshold
REQUIRED = [m[0] for m in METRICS] + ["min_metrics_failing", "bootstrap_resamples", "seed"]


class GateError(Exception):
    pass


def load_thresholds(path):
    try:
        with open(path) as f:
            t = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise GateError(f"cannot read thresholds file {path}: {e}")
    missing = [k for k in REQUIRED if k not in t]
    if missing:
        raise GateError(f"thresholds file {path} is missing required key(s): {', '.join(missing)}")
    for k in REQUIRED:
        if not isinstance(t[k], (int, float)) or isinstance(t[k], bool):
            raise GateError(f"threshold '{k}' must be a number, got {t[k]!r}")
    if t["min_metrics_failing"] not in (1, 2, 3):
        raise GateError("min_metrics_failing must be 1, 2 or 3")
    return t


def find_capture(name, dirs):
    if os.path.exists(name):
        return name
    for d in dirs:
        p = os.path.join(d, name + ".json.gz")
        if os.path.exists(p):
            return p
    raise GateError(f"capture '{name}' not found (looked at the path itself and in: {', '.join(os.path.abspath(d) for d in dirs)})")


def evaluate(ref, cand, th):
    """ref, cand: loaded captures. Returns {'metrics': {...}, 'failed': n, 'verdict': 'PASS'|'FAIL'}."""
    try:
        means = pm_.passage_means(pm_.per_passage(ref, cand))
    except AssertionError as e:
        raise GateError(f"captures are not comparable: {e}")
    n = len(means["kl"])
    point = pm_.overall(means, np.arange(n)[None, :])
    boot = pm_.overall(means, pm_.bootstrap_idx(n, n, int(th["bootstrap_resamples"]), np.random.default_rng(int(th["seed"]))))
    out = {}
    for key, col, _, scale in METRICS:
        v = float(point[col][0]) * scale; lo, hi = np.percentile(boot[col] * scale, [2.5, 97.5])
        out[key] = {"value": v, "ci_low": float(lo), "ci_high": float(hi), "threshold": float(th[key]), "fail": bool(v > th[key])}
    failed = sum(m["fail"] for m in out.values())
    return {"metrics": out, "failed": failed, "passages": n, "verdict": "FAIL" if failed >= th["min_metrics_failing"] else "PASS"}


def report(res, ref_name, cand_name, th):
    print(f"GATE  reference={ref_name}  candidate={cand_name}  ({res['passages']} passages)")
    print(f"{'metric':24s}{'value':>11s}   {'95% bootstrap CI':24s}{'threshold':>10s}  result")
    for key, _, label, _ in METRICS:
        m = res["metrics"][key]
        print(f"{label:24s}{m['value']:11.5f}   [{m['ci_low']:.5f}, {m['ci_high']:.5f}]".ljust(61) + f"{m['threshold']:10.5f}  {'FAIL' if m['fail'] else 'PASS'}")
    print(f"rule: gate fails when at least {int(th['min_metrics_failing'])} of 3 metrics fail (decision uses the point value) -> {res['failed']} failed")
    print(f"VERDICT: {res['verdict']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("reference"); p.add_argument("candidate")
    p.add_argument("--thresholds", required=True)
    p.add_argument("--captures-dir", action="append", help="extra place to look up labels (repeatable)")
    a = p.parse_args(argv)
    try:
        th = load_thresholds(a.thresholds)
        dirs = (a.captures_dir or []) + DEFAULT_DIRS
        rp, cp = find_capture(a.reference, dirs), find_capture(a.candidate, dirs)
        res = evaluate(pm_.lc.load_capture(rp), pm_.lc.load_capture(cp), th)
    except GateError as e:
        print(f"GATE ERROR: {e}", file=sys.stderr)
        return 2
    report(res, a.reference, a.candidate, th)
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
