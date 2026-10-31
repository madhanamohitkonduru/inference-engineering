"""Chunked-prefill stall experiment.
Usage: python chunked_prefill_test.py <label> <mode>      mode = baseline | intruder

6 'victim' requests decode ~1000 tokens each (streamed). Once every victim has produced 50 tokens,
mode 'intruder' sends ONE long prompt (3,500 tokens, random nonce at the start so nothing can be cached).
For every victim we record the arrival time of each streamed chunk (~1 token) and compute the gaps
between chunks. Gaps are taken from token 50 to 50-before-the-end, so start-up and wind-down are excluded.
mode 'baseline' is the same 6 victims with no intruder, to show normal gaps and noise.
"""
import json, random, statistics, sys, threading, time, requests
from transformers import AutoTokenizer

BASE = "http://localhost:8000"
MODEL = "Qwen/Qwen3-0.6B"
label = sys.argv[1] if len(sys.argv) > 1 else "run"
mode = sys.argv[2] if len(sys.argv) > 2 else "baseline"
assert mode in ("baseline", "intruder")
N_VICTIMS, VICTIM_TOKENS, STEADY_AFTER = 6, 1000, 50
INTRUDER_TOKENS, INTRUDER_OUT = 3500, 10
tok = AutoTokenizer.from_pretrained(MODEL)


def stream(prompt_ids, max_tokens, arrivals, sent_at=None):
    """Send one streamed completion; append the arrival time of every data chunk to `arrivals`."""
    body = {"model": MODEL, "prompt": prompt_ids, "max_tokens": max_tokens, "temperature": 0,
            "ignore_eos": True, "stream": True}
    if sent_at is not None:
        sent_at.append(time.perf_counter())
    with requests.post(BASE + "/v1/completions", json=body, stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines(chunk_size=None):
            if line.startswith(b"data: ") and line != b"data: [DONE]":
                arrivals.append(time.perf_counter())


def intruder_prompt():
    nonce = [random.randint(1000, 100000) for _ in range(8)]       # new every call: nothing can be cached
    filler = tok.encode("The quick brown fox jumps over the lazy dog. " * 600)
    p = nonce + filler[: INTRUDER_TOKENS - len(nonce)]
    assert len(p) == INTRUDER_TOKENS
    return p


# --- warm-up: one short request, plus (intruder mode) one throw-away long prompt so first-use costs are not measured
stream(tok.encode("Hello"), 5, [])
if mode == "intruder":
    stream(intruder_prompt(), 2, [])

victim_prompt = tok.encode("Write a very long story about a robot who learns to cook.")
arrivals = [[] for _ in range(N_VICTIMS)]
threads = [threading.Thread(target=stream, args=(victim_prompt, VICTIM_TOKENS, arrivals[i])) for i in range(N_VICTIMS)]
t0 = time.perf_counter()
for th in threads:
    th.start()

intruder_sent, intruder_arrivals = [], []
if mode == "intruder":
    while min(len(a) for a in arrivals) < STEADY_AFTER:          # wait until every victim is decoding steadily
        time.sleep(0.005)
    th = threading.Thread(target=stream, args=(intruder_prompt(), INTRUDER_OUT, intruder_arrivals, intruder_sent))
    th.start(); threads.append(th)
for th in threads:
    th.join()

# --- analysis
gaps = []                                   # (gap_ms, seconds_after_intruder_sent_or_None)
for a in arrivals:
    for i in range(STEADY_AFTER, len(a) - STEADY_AFTER):
        g = (a[i] - a[i - 1]) * 1000
        gaps.append((g, (a[i] - intruder_sent[0]) if intruder_sent else None))
vals = sorted(g for g, _ in gaps)
median = statistics.median(vals)
p99 = vals[int(0.99 * len(vals)) - 1]
worst = max(gaps)
top5 = sorted(gaps, reverse=True)[:5]
chunk_counts = [len(a) for a in arrivals]
ttft = (intruder_arrivals[0] - intruder_sent[0]) * 1000 if intruder_arrivals else None

print(f"\n=== {label}: mode {mode}, {N_VICTIMS} victims x {VICTIM_TOKENS} tokens ===")
print(f"chunks per victim: {chunk_counts}   (should be close to {VICTIM_TOKENS}: one chunk ~ one token)")
print(f"gaps measured: {len(vals)}   median {median:.2f} ms   p99 {p99:.2f} ms   WORST {worst[0]:.2f} ms")
print("5 largest gaps (ms, seconds after the intruder was sent):", [(round(g, 1), None if s is None else round(s, 3)) for g, s in top5])
if ttft is not None:
    print(f"intruder TTFT: {ttft:.1f} ms")
json.dump({"label": label, "mode": mode, "median_ms": median, "p99_ms": p99, "worst_ms": worst[0],
           "top5": top5, "chunk_counts": chunk_counts, "intruder_ttft_ms": ttft,
           "arrivals_s": [[x - t0 for x in a] for a in arrivals]}, open(f"result_{label}.json", "w"))
