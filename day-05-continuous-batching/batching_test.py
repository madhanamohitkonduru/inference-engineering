"""Send 1 long + 3 short requests at (almost) the same time; print when each one finishes."""
import json, sys, threading, time, requests

URL = "http://localhost:8000/v1/completions"
MODEL = "Qwen/Qwen3-0.6B"
# (name, max_tokens). The long one is sent first so it takes the seat first.
JOBS = [("long", 400), ("short1", 20), ("short2", 20), ("short3", 20)]
results = {}

def run(name, max_tokens, t0):
    body = {"model": MODEL, "prompt": "Write a long story about a robot who learns to cook.",
            "max_tokens": max_tokens, "temperature": 0,
            "ignore_eos": True}  # force exactly max_tokens tokens, so lengths are fair
    r = requests.post(URL, json=body, timeout=600)
    r.raise_for_status()
    results[name] = time.time() - t0

t0 = time.time()
threads = []
for name, mt in JOBS:
    th = threading.Thread(target=run, args=(name, mt, t0))
    th.start(); threads.append(th)
    time.sleep(0.05)  # tiny gap so the order is long, short1, short2, short3
for th in threads: th.join()

label = sys.argv[1] if len(sys.argv) > 1 else "run"
print(f"\n=== {label} ===")
for name, _ in JOBS:
    print(f"{name:7s} finished at {results[name]:6.2f} s")
json.dump(results, open(f"result_{label}.json", "w"), indent=2)
