import json, re, glob, os
import pandas as pd

pattern = re.compile(r"^(?P<config>.+)_run(?P<run>\d+)_(?P<workload>copy|free)_c(?P<concurrency>\d+)\.json$")
rows = []
for path in sorted(glob.glob("day-09-speculative-decoding/results/*.json")):
    name = os.path.basename(path)
    meta = pattern.match(name).groupdict()      # {'config': 'base', 'run': '1', ...}
    data = json.load(open(path))
    for i, r in enumerate(data["requests"]):
        rows.append({**meta, "request_index": i, **r})
df = pd.DataFrame(rows)
df["run"] = df["run"].astype(int)
df["concurrency"] = df["concurrency"].astype(int)
print(df.shape)
print(df.groupby(["config", "workload", "concurrency"]).size())
g = df.groupby(["config", "workload", "concurrency"])["tpot_ms"]
tpot = g.agg(
    n="size",
    mean="mean",
    p50=lambda s: s.quantile(0.50),
    p95=lambda s: s.quantile(0.95),
    p99=lambda s: s.quantile(0.99),
).round(3)
print(tpot)
df.to_parquet("day-10-goodput-analysis/day9_requests.parquet", index=False)