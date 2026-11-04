#!/usr/bin/env python3
"""Correctness harness, part 1: capture prompt logprobs from a vLLM server, then compare two captures.

  capture: send fixed passages to a running server, save per-position logprobs (teacher forcing).
  compare: print flip rate, mean |logprob diff|, approximate KL (top-20) and perplexity for two captures.

Usage (from the repo root):
  python day-11-correctness-harness/logprob_compare.py capture bf16_c1_run1 --concurrency 1
  python day-11-correctness-harness/logprob_compare.py compare bf16_c1_run1 fp8_c1
"""
import argparse, gzip, json, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "Qwen/Qwen3-0.6B"
TOPK = 20


# ----------------------------------------------------------------- passages
def build_passages(path, model=MODEL, n=50, length=512):
    """Cut n passages of `length` tokens from the WikiText-2 test set (real Wikipedia text, not model-written)."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    f = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")
    text = "".join(pd.read_parquet(f)["text"])
    tok = AutoTokenizer.from_pretrained(model)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    step = (len(ids) // length) // n                  # spread the passages evenly over the whole test set
    chunks = [ids[i * step * length:(i * step + 1) * length] for i in range(n)]
    data = {"source": "WikiText-2 raw test split (HF dataset Salesforce/wikitext, wikitext-2-raw-v1)",
            "tokenizer": model, "length": length,
            "passages": [{"ids": c, "pieces": [tok.decode([i]) for i in c]} for c in chunks]}
    with open(path, "w") as fh:
        json.dump(data, fh)
    return data


def load_passages(path):
    if not os.path.exists(path):
        print(f"building {path} (needs network: downloads WikiText-2 and the tokenizer)")
        return build_passages(path)
    with open(path) as fh:
        return json.load(fh)


# ------------------------------------------------------------------ capture
def query(url, model, ids, topk=TOPK):
    """One request. prompt = token ids (so the true next token at every position is known exactly)."""
    import requests
    body = {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": topk}
    r = requests.post(url + "/v1/completions", json=body, timeout=900)
    r.raise_for_status()
    return r.json()


def parse(ids, resp, topk=TOPK):
    """Raw response -> per position (1..n-1): true token logprob, top-1 id, top-k [id, logprob] list.
    prompt_logprobs[0] is None (the first token has nothing before it), so position 0 is skipped."""
    pl = resp["choices"][0]["prompt_logprobs"]
    assert len(pl) == len(ids) and pl[0] is None, "unexpected prompt_logprobs shape"
    true_lp, top1, top = [], [], []
    for pos in range(1, len(ids)):
        d = pl[pos]                                       # {token_id(str): {"logprob":..,"rank":..,"decoded_token":..}}
        true_lp.append(d[str(ids[pos])]["logprob"])       # the real next token is always included
        items = sorted(((int(k), v["logprob"]) for k, v in d.items()), key=lambda x: -x[1])[:topk]
        top1.append(items[0][0])
        top.append([[i, lp] for i, lp in items])
    return {"true_ids": ids[1:], "true_lp": true_lp, "top1": top1, "top": top}


def cmd_capture(a):
    data = load_passages(os.path.join(HERE, "passages.json"))
    passages = data["passages"]
    if a.peek:                                            # look at one raw response first
        resp = query(a.url, a.model, passages[0]["ids"])
        pl = resp["choices"][0]["prompt_logprobs"]
        print("top-level keys:", list(resp.keys()), "| choice keys:", list(resp["choices"][0].keys()))
        print("len(prompt_logprobs) =", len(pl), "| prompt_logprobs[0] =", pl[0])
        print("prompt_logprobs[1] =", json.dumps(pl[1], indent=1)[:1500])
        print("entries at position 1:", len(pl[1]), "(top-20 plus the true token if it was not in the top 20)")
        return
    t0 = time.time()
    done = [0]

    def work(p):
        out = parse(p["ids"], query(a.url, a.model, p["ids"]))
        done[0] += 1
        if done[0] % 10 == 0:
            print(f"  {done[0]}/{len(passages)} passages ({time.time() - t0:.0f}s)", flush=True)
        return out

    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:   # concurrency 1 = one at a time
        results = list(ex.map(work, passages))
    cap = {"label": a.label,
           "meta": {"model": a.model, "concurrency": a.concurrency, "note": a.note, "topk": TOPK,
                    "passages_source": data["source"], "n_passages": len(passages), "seconds": round(time.time() - t0, 1),
                    "captured_at_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())},
           "passages": results}
    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, a.label + ".json.gz")
    with gzip.open(path, "wt") as fh:
        json.dump(cap, fh)
    print(f"saved {path} ({os.path.getsize(path) / 1e6:.1f} MB, {time.time() - t0:.0f}s)")


# ------------------------------------------------------------------ compare
def load_capture(path):
    with gzip.open(path, "rt") as fh:
        cap = json.load(fh)
    return {"label": cap["label"], "meta": cap["meta"],
            "true_ids": np.array([t for p in cap["passages"] for t in p["true_ids"]]),
            "true_lp": np.array([t for p in cap["passages"] for t in p["true_lp"]]),
            "top1": np.array([t for p in cap["passages"] for t in p["top1"]]),
            "top": np.array([t for p in cap["passages"] for t in p["top"]])}   # shape (positions, 20, 2): id, logprob


def approx_kl_one(top_a, top_b):
    """KL(A || B) from two top-20 lists. Approximation: work on the union of both lists; a token missing from one
    side gets that side's smallest top-20 probability; renormalize each side to sum to 1 over the union."""
    pa = dict(zip(top_a[:, 0].astype(int), np.exp(top_a[:, 1])))
    pb = dict(zip(top_b[:, 0].astype(int), np.exp(top_b[:, 1])))
    fa, fb = min(pa.values()), min(pb.values())
    keys = sorted(set(pa) | set(pb))
    qa = np.array([pa.get(k, fa) for k in keys]); qa /= qa.sum()
    qb = np.array([pb.get(k, fb) for k in keys]); qb /= qb.sum()
    return float(np.sum(qa * np.log(qa / qb)))


def per_position(a, b):
    """Per-position differences between two loaded captures (the same passages, same order)."""
    assert a["true_ids"].shape == b["true_ids"].shape and (a["true_ids"] == b["true_ids"]).all(), \
        "captures were not made from the same passages"
    return {"abs_diff": np.abs(a["true_lp"] - b["true_lp"]),
            "flip": a["top1"] != b["top1"],
            "kl": np.array([approx_kl_one(x, y) for x, y in zip(a["top"], b["top"])])}


def perplexity(cap):
    return math.exp(-float(np.mean(cap["true_lp"])))


def summarize(a, b):
    pp = per_position(a, b)
    ppl_a, ppl_b = perplexity(a), perplexity(b)
    return {"positions": int(len(pp["flip"])),
            "mean_abs_logprob_diff": float(pp["abs_diff"].mean()),
            "max_abs_logprob_diff": float(pp["abs_diff"].max()),
            "top1_agreement_pct": 100 * (1 - float(pp["flip"].mean())),
            "flip_rate_pct": 100 * float(pp["flip"].mean()),
            "approx_kl_mean": float(pp["kl"].mean()),
            "perplexity_a": ppl_a, "perplexity_b": ppl_b, "perplexity_diff": ppl_b - ppl_a}


def cmd_compare(a):
    ca = load_capture(os.path.join(a.outdir, a.label_a + ".json.gz"))
    cb = load_capture(os.path.join(a.outdir, a.label_b + ".json.gz"))
    s = summarize(ca, cb)
    print(f"A = {a.label_a}   B = {a.label_b}   ({s['positions']} positions)")
    print(f"  mean |true-token logprob diff| : {s['mean_abs_logprob_diff']:.6f}   (max {s['max_abs_logprob_diff']:.4f})")
    print(f"  top-1 agreement                : {s['top1_agreement_pct']:.3f} %   (flip rate {s['flip_rate_pct']:.3f} %)")
    print(f"  approx KL from top-20 (mean)   : {s['approx_kl_mean']:.8f}   (approximation, see docstring)")
    print(f"  perplexity A / B / B - A       : {s['perplexity_a']:.4f} / {s['perplexity_b']:.4f} / {s['perplexity_diff']:+.4f}")
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default=os.path.join(HERE, "captures"))
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("label", nargs="?", default="peek")
    c.add_argument("--url", default="http://localhost:8000")
    c.add_argument("--model", default=MODEL)
    c.add_argument("--concurrency", type=int, default=1)
    c.add_argument("--note", default="", help="server flags, for the record")
    c.add_argument("--peek", action="store_true", help="print one raw response and exit")
    c.set_defaults(fn=cmd_capture)
    d = sub.add_parser("compare")
    d.add_argument("label_a"); d.add_argument("label_b")
    d.set_defaults(fn=cmd_compare)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
