import gzip, json, glob, os, re
import numpy as np
import pandas as pd

def load(path):
    with gzip.open(path, "rt") as f:
        d = json.load(f)
    ttft = np.array(d["ttfts"]) * 1000       # seconds -> ms
    lat = np.array(d["latencies"]) * 1000
    n_out = np.array(d["output_lens"])
    tpot = (lat - ttft) / (n_out - 1)         # time after the first token / tokens after the first
    return d, ttft, tpot

rows_req, rows_file = [], []
for cache in ["sweep_cacheoff", "sweep_cacheon"]:
    for path in sorted(glob.glob(f"day-10-goodput-analysis/{cache}/*.json.gz")):
        m = re.match(r"run(\d+)_c(\d+)\.json\.gz", os.path.basename(path))
        run, c = int(m.group(1)), int(m.group(2))
        d, ttft, tpot = load(path)
        rows_file.append({"cache": cache, "run": run, "concurrency": c, "prompts": d["num_prompts"],
                          "duration_s": d["duration"], "out_tok_s": d["output_throughput"],
                          "req_goodput": d["request_goodput"]})
        rows_req.append(pd.DataFrame({"cache": cache, "run": run, "concurrency": c,
                                      "idx": np.arange(len(ttft)),
                                      "ttft_ms": ttft, "tpot_ms": tpot}))
req = pd.concat(rows_req, ignore_index=True)     # one row per request
files = pd.DataFrame(rows_file)                  # one row per benchmark run
print(req.shape)
print(files.groupby(["cache", "concurrency"]).size())

off = req[req.cache == "sweep_cacheoff"]
thr = files[files.cache == "sweep_cacheoff"].groupby("concurrency")["out_tok_s"].mean().round(0)

def pcts(col):
    g = off.groupby("concurrency")[col]
    return pd.DataFrame({f"{col}_p50": g.quantile(0.50),
                         f"{col}_p95": g.quantile(0.95),
                         f"{col}_p99": g.quantile(0.99)})

table = pd.concat([thr.rename("tok_s"), pcts("ttft_ms"), pcts("tpot_ms")], axis=1).round(2)
print(table.to_string())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for a, col, limit, name in [(ax[0], "tpot_ms_p99", 10, "p99 TPOT (ms)"), (ax[1], "ttft_ms_p99", 200, "p99 TTFT (ms)")]:
    a.plot(table["tok_s"], table[col], "o-")
    for c, row in table.iterrows():
        a.annotate(f"c={c}", (row["tok_s"], row[col]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    a.axhline(limit, color="red", linestyle="--", label=f"SLO limit {limit} ms")
    a.set_xlabel("throughput (output tokens/s)")
    a.set_ylabel(name)
    a.set_yscale("log")
    a.legend()
fig.suptitle("Qwen3-0.6B on RTX 4090, 512 in / 128 out, prefix caching off (2 runs pooled)")
fig.tight_layout()
fig.savefig("day-10-goodput-analysis/latency_vs_throughput.png", dpi=150)
print("saved")

for c in [32, 64, 128, 256]:
    sub = off[off.concurrency == c]
    first, rest = sub[sub.idx < c], sub[sub.idx >= c]
    print(f"c={c:3d} first wave n={len(first):4d} p50={first.ttft_ms.median():7.0f} p99={first.ttft_ms.quantile(.99):7.0f} ms"
          f" | rest n={len(rest):5d} p50={rest.ttft_ms.median():7.0f} p99={rest.ttft_ms.quantile(.99):7.0f} ms")

req["ok"] = (req.ttft_ms <= 200) & (req.tpot_ms <= 10)
req.to_parquet("day-10-goodput-analysis/sweep_requests.parquet", index=False)
off = req[req.cache == "sweep_cacheoff"]
per_run = (off.groupby(["concurrency", "run"])["ok"].sum().rename("n_ok").reset_index()
           .merge(files[files.cache == "sweep_cacheoff"][["concurrency", "run", "prompts", "duration_s", "req_goodput"]],
                  on=["concurrency", "run"]))
per_run["my_goodput"] = per_run.n_ok / per_run.duration_s
per_run["req_tput"] = per_run.prompts / per_run.duration_s
gp = per_run.groupby("concurrency")[["req_tput", "my_goodput", "req_goodput"]].mean()
gp["pct_meeting_slo"] = off.groupby("concurrency")["ok"].mean() * 100
print(gp.round(2).to_string())

def capacity(ttft_limit, tpot_limit=10):
    ok = (off.ttft_ms <= ttft_limit) & (off.tpot_ms <= tpot_limit)
    frac = ok.groupby(off.concurrency).mean() * 100
    passing = frac[frac >= 99]
    return (int(passing.index.max()) if len(passing) else None), frac.round(1).to_dict()

for lim in [50, 100, 200, 400, 1000]:
    cap, frac = capacity(lim)
    print(f"TTFT <= {lim:4d} ms (TPOT <= 10): capacity c={cap}")

table.to_csv("day-10-goodput-analysis/latency_vs_throughput.csv")
gp.round(2).to_csv("day-10-goodput-analysis/goodput_table.csv")