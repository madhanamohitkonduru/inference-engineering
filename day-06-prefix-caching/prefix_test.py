"""Shared-prefix TTFT experiment.
Usage: python prefix_test.py <label> [prefix_tokens] [mode]
  mode = normal (default) | mid (one token in the middle of the prefix differs per request)
       | after (the unique question comes BEFORE the shared text)

Sends 1 warm-up request, then 10 requests ONE AT A TIME. Each request = the same shared prefix
(random 8-token nonce + filler, exactly prefix_tokens long) + a different short question.
TTFT = time from sending the request to the first streamed chunk.
Also reads vLLM's prefix-cache counters before/after to report how many tokens were cache hits.
"""
import json, random, re, sys, time, requests
from transformers import AutoTokenizer

BASE = "http://localhost:8000"
MODEL = "Qwen/Qwen3-0.6B"
label = sys.argv[1] if len(sys.argv) > 1 else "run"
PREFIX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
MODE = sys.argv[3] if len(sys.argv) > 3 else "normal"
assert MODE in ("normal", "mid", "after")
N = 10
tok = AutoTokenizer.from_pretrained(MODEL)
sess = requests.Session()   # reuse the connection so setup cost is not in every TTFT


def counters():
    """Return (queries, hits) in tokens, summed over the prefix-cache counters."""
    text = sess.get(BASE + "/metrics").text
    def total(name):
        return sum(float(v) for v in re.findall(rf"^{name}{{[^}}]*}} ([0-9.e+]+)$", text, re.M))
    return total("vllm:prefix_cache_queries_total"), total("vllm:prefix_cache_hits_total")


def ttft_ms(prompt_ids):
    body = {"model": MODEL, "prompt": prompt_ids, "max_tokens": 20, "temperature": 0,
            "ignore_eos": True, "stream": True}
    t0 = time.perf_counter()
    first = None
    with sess.post(BASE + "/v1/completions", json=body, stream=True, timeout=120) as r:
        r.raise_for_status()
        for line in r.iter_lines(chunk_size=None):
            if first is None and line.startswith(b"data: ") and line != b"data: [DONE]":
                first = time.perf_counter() - t0
    return first * 1000


# Shared prefix = random nonce (new every run, so every run starts cold) + filler, exactly PREFIX_TOKENS long.
nonce = [random.randint(1000, 100000) for _ in range(8)]
filler = tok.encode("The quick brown fox jumps over the lazy dog. " * 400)
prefix = nonce + filler[: PREFIX_TOKENS - len(nonce)]
assert len(prefix) == PREFIX_TOKENS, "filler too short for this prefix length"
questions = [tok.encode(f" Question {i}: what is {i} plus {i + 7}?") for i in range(N)]

def build(i):
    if MODE == "mid":                      # change ONE token in the middle; id is unique per request (valid: vocab is 151,936)
        p = list(prefix)
        p[PREFIX_TOKENS // 2] = 100000 + i
        return p + questions[i]
    if MODE == "after":                    # unique question first, shared text after it
        return questions[i] + prefix
    return prefix + questions[i]           # normal: shared prefix first


ttft_ms([random.randint(1000, 100000) for _ in range(20)])   # warm-up: unrelated short request
q0, h0 = counters()
results = [ttft_ms(build(i)) for i in range(N)]
q1, h1 = counters()

print(f"\n=== {label}: prefix {PREFIX_TOKENS} tokens, mode {MODE}, {N} requests, one at a time ===")
for i, v in enumerate(results, 1):
    print(f"request {i:2d}: TTFT {v:7.1f} ms")
print(f"cache counters during run: queries +{q1 - q0:.0f}, hits +{h1 - h0:.0f}")
json.dump({"label": label, "prefix_tokens": PREFIX_TOKENS, "mode": MODE, "ttft_ms": results,
           "queries": q1 - q0, "hits": h1 - h0}, open(f"result_{label}.json", "w"), indent=2)
