"""Day 14 Step 5: decode-time model for tensor parallelism.  python3 decode_model.py

step time = (weight bytes read + KV bytes read) / memory bandwidth              [per GPU, after splitting by TP]
          + (2 x layers) x (all-reduce latency + message bytes / link bandwidth) [only when TP > 1]
tokens/s per user = 1 / step time   (one new token per user per step)

ASSUMPTIONS, not measurements (Day 15 will measure them):
  all-reduce latency: NVLink 10 us, PCIe 30 us. I found no sourced small-message number, so these are guesses based on
  "NVLink is a direct GPU-to-GPU link; PCIe goes through more hops and is several times slower for tiny messages".
  link bandwidth, per direction: NVLink 450 GB/s (HYPOTHESIS: half of NVIDIA's quoted 900 GB/s), PCIe 4.0 x16 31.5 GB/s.
Simplifications: ring all-reduce factor 2(N-1)/N ignored (at most 2x on a term that is tiny); the same latency is used for
every TP size (real latency grows with more GPUs); compute time is ignored (decode is memory-bound).
"""
import csv
from pathlib import Path

import memory_plan as mp

OUT = Path(__file__).parent / "tables"
OUT.mkdir(exist_ok=True)

LINKS = {"nvlink": dict(latency=10e-6, bw=450e9), "pcie4": dict(latency=30e-6, bw=31.5e9)}


def step_time(model, gpu, tp, link=None, context=1024, batch=1, latency=None, link_bw=None, util=0.9):
    r = mp.plan(model, gpu, tp, util=util, context=context)
    m = mp.MODELS[model]
    bandwidth = mp.GPUS[gpu]["bandwidth"]
    read_bytes = r["weights_read_gpu"] + batch * context * r["kv_per_token_gpu"]   # weights once, KV for every user
    memory_time = read_bytes / bandwidth
    comm = 0.0
    if tp > 1:
        lk = LINKS[link]
        lat = lk["latency"] if latency is None else latency
        bw = lk["bw"] if link_bw is None else link_bw
        message = batch * m["hidden"] * 2          # one hidden vector per user, BF16
        comm = 2 * m["layers"] * (lat + message / bw)
    return memory_time, comm


def tps(model, gpu, tp, link=None, **kw):
    mt, ct = step_time(model, gpu, tp, link, **kw)
    return 1.0 / (mt + ct)


if __name__ == "__main__":
    rows = []
    print("Qwen3-8B, context 1,024, tokens/s per user (speedup vs TP=1 in brackets)\n")
    for batch in (1, 32):
        print(f"batch {batch}")
        print(f"{'GPU':8} {'link':7} " + " ".join(f"{'TP='+str(t):>16}" for t in (1, 2, 4, 8)))
        for gpu in ("rtx4090", "h100-80"):
            for link in ("nvlink", "pcie4"):
                base = tps("qwen3-8b", gpu, 1, link, batch=batch)
                cells = []
                for tp in (1, 2, 4, 8):
                    v = tps("qwen3-8b", gpu, tp, link, batch=batch)
                    mt, ct = step_time("qwen3-8b", gpu, tp, link, batch=batch)
                    rows.append(dict(model="qwen3-8b", gpu=gpu, link=link, batch=batch, tp=tp, tokens_per_s_per_user=round(v, 1),
                                     speedup_vs_tp1=round(v / base, 2), memory_ms=round(mt * 1e3, 3), allreduce_ms=round(ct * 1e3, 3)))
                    cells.append(f"{v:8.1f} ({v/base:4.2f}x)")
                note = "  <- 4090 has no NVLink: hypothetical" if gpu == "rtx4090" and link == "nvlink" else ""
                print(f"{gpu:8} {link:7} " + " ".join(f"{c:>16}" for c in cells) + note)
        print()
    with open(OUT / "decode_model.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # Sensitivity at TP=2 on an H100 (batch 1 and 32): vary one assumption at a time
    print("Sensitivity: qwen3-8b on h100-80, TP=2, NVLink assumptions, tokens/s per user")
    sens = []
    for batch in (1, 32):
        base = tps("qwen3-8b", "h100-80", 2, "nvlink", batch=batch)
        print(f"batch {batch}: baseline (10 us, 450 GB/s) = {base:.1f}")
        for lat in (2e-6, 5e-6, 10e-6, 20e-6, 40e-6, 80e-6):
            v = tps("qwen3-8b", "h100-80", 2, "nvlink", batch=batch, latency=lat)
            sens.append(dict(batch=batch, vary="latency_us", value=lat * 1e6, tokens_per_s=round(v, 1), vs_baseline=round(v / base, 3)))
            print(f"  latency {lat*1e6:5.0f} us            -> {v:7.1f} ({v/base:5.3f} of baseline)")
        for bw in (30e9, 100e9, 450e9, 1800e9):
            v = tps("qwen3-8b", "h100-80", 2, "nvlink", batch=batch, link_bw=bw)
            sens.append(dict(batch=batch, vary="link_bw_GBps", value=bw / 1e9, tokens_per_s=round(v, 1), vs_baseline=round(v / base, 3)))
            print(f"  link bandwidth {bw/1e9:6.0f} GB/s    -> {v:7.1f} ({v/base:5.3f} of baseline)")
    with open(OUT / "sensitivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sens[0])); w.writeheader(); w.writerows(sens)
