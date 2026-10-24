#!/usr/bin/env python3
"""Summarize a PyTorch-profiler Chrome trace: top GPU kernels, attention/GEMM/other split, GPU idle gaps.

Usage: python3 trace_top_kernels.py TRACE.json[.gz] [--top 5]
Standard library only. Output is markdown, ready to paste into a README.
"""
import argparse, collections, gzip, json, re

ATTN = re.compile(r"flash|attn|attention|fmha|mha|paged", re.I)
GEMM = re.compile(r"gemm|cutlass|cublas|nvjet|xmma|matmul|wgmma|ampere_|ada_|sm\d+_", re.I)

def classify(name: str) -> str:
    if "cache" in name.lower():          # e.g. reshape_and_cache: KV write, not attention math
        return "other"
    if ATTN.search(name):
        return "attention"
    if GEMM.search(name):
        return "GEMM"
    return "other"

def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        data = json.load(f)
    return data["traceEvents"] if isinstance(data, dict) else data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()

    ev = [e for e in load(a.trace)
          if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
    if not ev:
        raise SystemExit("No GPU kernel events found (looked for cat=kernel/gpu_memcpy/gpu_memset).")

    tot = collections.defaultdict(float); cnt = collections.Counter()
    for e in ev:
        tot[e["name"]] += e["dur"]; cnt[e["name"]] += 1
    gpu_time = sum(tot.values())

    # busy time = union of kernel intervals across streams; idle = gaps inside the observed window
    iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev)
    busy, gaps, cur_s, cur_e = 0.0, [], iv[0][0], iv[0][1]
    for s, t in iv[1:]:
        if s > cur_e:
            busy += cur_e - cur_s; gaps.append(s - cur_e); cur_s, cur_e = s, t
        else:
            cur_e = max(cur_e, t)
    busy += cur_e - cur_s
    span = max(t for _, t in iv) - iv[0][0]
    idle = span - busy

    print(f"Trace: {a.trace}\n")
    print(f"- GPU events: {len(ev):,}; summed kernel time: {gpu_time/1e3:,.1f} ms; observed window: {span/1e3:,.1f} ms")
    print(f"- GPU busy (union of intervals): {busy/1e3:,.1f} ms = {100*busy/span:.1f}% of window; idle: {idle/1e3:,.1f} ms")
    if gaps:
        gs = sorted(gaps)
        big = [g for g in gaps if g > 20]
        print(f"- Gaps between kernels: {len(gaps):,}; median {gs[len(gs)//2]:.1f} us; p99 {gs[int(len(gs)*0.99)]:.1f} us; "
              f"gaps > 20 us: {len(big):,} totalling {sum(big)/1e3:,.1f} ms\n")

    print(f"| Rank | Kernel | Class | Calls | Total ms | % of kernel time |\n|---|---|---|---|---|---|")
    for i, (n, t) in enumerate(sorted(tot.items(), key=lambda x: -x[1])[:a.top], 1):
        short = (n[:70] + "...") if len(n) > 70 else n
        print(f"| {i} | `{short}` | {classify(n)} | {cnt[n]:,} | {t/1e3:,.2f} | {100*t/gpu_time:.1f}% |")

    split = collections.defaultdict(float)
    for n, t in tot.items():
        split[classify(n)] += t
    print("\n| Class | Total ms | % of kernel time |\n|---|---|---|")
    for k in ("attention", "GEMM", "other"):
        print(f"| {k} | {split[k]/1e3:,.2f} | {100*split[k]/gpu_time:.1f}% |")

if __name__ == "__main__":
    main()
