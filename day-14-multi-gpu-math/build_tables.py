"""Day 14 Step 5: fit table, capacity table and doubling check from memory_plan.py. python3 build_tables.py"""
import csv
from pathlib import Path

import memory_plan as mp

OUT = Path(__file__).parent / "tables"
OUT.mkdir(exist_ok=True)
MODELS = ["qwen3-8b", "qwen3-32b", "qwen2.5-72b"]
GPUS = ["rtx4090", "a100-80", "h100-80", "h200"]
TPS = [1, 2, 4, 8]
CONTEXT, UTIL = 8192, 0.9          # 8,192-token sequences; vLLM default utilization is 0.9 (ASSUMPTION: same on all GPUs)
MARK = {"doesn't fit": "no", "fits": "FITS"}


def mark(r):
    v = r["verdict"]
    return "no" if v == "doesn't fit" else ("FITS" if v == "fits" else "tight")


rows, cap_rows = [], []
for m in MODELS:
    for g in GPUS:
        for tp in TPS:
            r = mp.plan(m, g, tp, util=UTIL, context=CONTEXT)
            rows.append(dict(model=m, gpu=g, tp=tp, verdict=r["verdict"], mark=mark(r), weights_gib=round(r["weights_gpu"] / mp.GIB, 2),
                             free_kv_gib=round(max(r["free_gib"], 0), 2), kv_tokens=r["kv_tokens"], seqs_8192=r["kv_seqs"]))
with open(OUT / "fit_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

lines = ["Fit table (8,192-token sequences, gpu_memory_utilization 0.9, activation 2.0 GiB assumed). "
         "no = doesn't fit; tight = fits but fewer than 4 sequences; FITS = at least 4 sequences. Cell = mark (sequences of 8,192).", ""]
lines.append("| Model | GPU | TP=1 | TP=2 | TP=4 | TP=8 |")
lines.append("|---|---|---|---|---|---|")
for m in MODELS:
    for g in GPUS:
        cells = []
        for tp in TPS:
            r = next(x for x in rows if x["model"] == m and x["gpu"] == g and x["tp"] == tp)
            cells.append("no" if r["mark"] == "no" else f"{r['mark']} ({r['seqs_8192']})")
        lines.append(f"| {m} | {g} | " + " | ".join(cells) + " |")
(OUT / "fit_table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# Doubling check: Qwen3-32B on an 80 GB GPU is the case where TP=1 barely fits.
print("\nDoubling check: qwen3-32b on h100-80 (TP=1 barely fits)")
dc = []
base = None
for tp in (1, 2, 4):
    r = mp.plan("qwen3-32b", "h100-80", tp, util=UTIL, context=CONTEXT)
    base = base or r["kv_tokens"]
    dc.append(dict(tp=tp, weights_gib=round(r["weights_gpu"] / mp.GIB, 2), free_kv_gib=round(r["free_gib"], 2),
                   kv_heads_per_gpu=r["kv_heads_gpu"], kv_bytes_per_token_per_gpu=int(r["kv_per_token_gpu"]),
                   kv_tokens=r["kv_tokens"], seqs_8192=r["kv_seqs"], vs_tp1=round(r["kv_tokens"] / base, 2)))
    print(dc[-1])
with open(OUT / "doubling_check.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(dc[0])); w.writeheader(); w.writerows(dc)
