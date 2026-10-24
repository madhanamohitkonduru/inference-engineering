#!/usr/bin/env python3
"""Summarize an Nsight Systems SQLite export: GPU busy time, idle gaps, and idle time per decode step.

Usage:
  nsys export --type sqlite --force-overwrite=true -o REPORT.sqlite REPORT.nsys-rep
  python3 nsys_gaps.py REPORT.sqlite [--step-kernel temperature]

A "step" is counted as one launch of a kernel whose name contains --step-kernel (default: "temperature",
a sampling kernel that runs once per engine step). Standard library only; output is markdown.
"""
import argparse, sqlite3

def tables(con):
    return {r[0] for r in con.execute("select name from sqlite_master where type='table'")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--step-kernel", default="temperature")
    a = ap.parse_args()
    con = sqlite3.connect(a.db)
    t = tables(con)
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in t:
        raise SystemExit(f"No kernel table found. Tables: {sorted(t)[:20]}")

    cols = {r[1] for r in con.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    namecol = "demangledName" if "demangledName" in cols else "shortName"
    kern = con.execute(
        f"select k.start, k.end, coalesce(s.value, cast(k.{namecol} as text)) "
        f"from CUPTI_ACTIVITY_KIND_KERNEL k left join StringIds s on s.id = k.{namecol}").fetchall()
    iv = [(s, e) for s, e, _ in kern]
    for tbl in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if tbl in t:
            iv += con.execute(f"select start, end from {tbl}").fetchall()
    if not iv:
        raise SystemExit("No GPU activity found.")
    iv.sort()

    busy, gaps, cs, ce = 0, [], iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s > ce:
            busy += ce - cs; gaps.append(s - ce); cs, ce = s, e
        else:
            ce = max(ce, e)
    busy += ce - cs
    span = max(e for _, e in iv) - iv[0][0]
    steps = sum(1 for _, _, n in kern if n and a.step_kernel.lower() in n.lower())
    ms = lambda ns: ns / 1e6

    print(f"Report: {a.db}\n")
    print(f"- GPU activities: {len(iv):,}; window: {ms(span):,.1f} ms; busy: {ms(busy):,.1f} ms ({100*busy/span:.1f}%); idle: {ms(span-busy):,.1f} ms")
    if gaps:
        g = sorted(gaps); big = [x for x in gaps if x > 20_000]
        print(f"- Gaps: {len(gaps):,}; median {g[len(g)//2]/1e3:.2f} us; p99 {g[int(len(g)*0.99)]/1e3:.1f} us; "
              f"gaps > 20 us: {len(big):,} totalling {ms(sum(big)):,.1f} ms")
    if steps:
        print(f"- Steps counted (kernels matching '{a.step_kernel}'): {steps:,}")
        print(f"- Per step: window {ms(span)/steps:.3f} ms; busy {ms(busy)/steps:.3f} ms; idle {ms(span-busy)/steps:.3f} ms")
    else:
        print(f"- No kernels matched '{a.step_kernel}'; per-step numbers unavailable (try --step-kernel with a name from the kernel list).")
    print(f"- Kernel launches per step: {len(kern)/steps:,.0f}" if steps else "")

if __name__ == "__main__":
    main()
