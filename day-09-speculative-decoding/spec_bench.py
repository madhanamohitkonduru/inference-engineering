"""Speculative-decoding benchmark: copy-heavy vs free-writing prompts at a chosen concurrency.
Usage: python spec_bench.py <label> <copy|free> <concurrency> [num_requests] [max_tokens]
  label  = which server is running (e.g. base, ngram_k4); only used in the output file name
Streams /v1/completions (temperature 0, ignore_eos), measures TTFT / TPOT / throughput, and reads the
speculative-decoding counters from /metrics before and after the run (acceptance rate, mean acceptance length).
Saves results/<label>_<workload>_c<concurrency>.json"""
import json, os, re, sys, time, requests
from concurrent.futures import ThreadPoolExecutor

URL = "http://localhost:8000"
MODEL = "Qwen/Qwen3-0.6B"

PASSAGE = (
    "The harbor town of Marrow Bay sits where the river meets the sea, and for three hundred years its fortunes "
    "have risen and fallen with the tide. In spring the fishing boats leave before dawn, their lanterns trailing "
    "yellow lines across the black water, and return at dusk loaded with silver herring that the women of the "
    "quayside gut, salt and pack into barrels stamped with the name of the town. In summer the cruise ships "
    "anchor beyond the breakwater and the narrow streets fill with visitors who photograph the painted shutters, "
    "buy postcards from the chandler's shop and queue for the lighthouse, where a keeper named Odalys Venn "
    "still winds the great brass clockwork by hand every evening. Autumn brings storms from the northwest; the "
    "harbor master rings a bell at the first warning, shopkeepers lower their awnings, and children are called "
    "indoors. Winter is quiet. Smoke rises from the chimneys of the old customs house, nets are mended in "
    "kitchens, and the only sound on the stone mole is the slow knock of rigging against masts. Every year, on "
    "the shortest day, the townspeople carry lamps down to the water and set them afloat, one for each boat "
    "lost since the founding of the port, until the whole bay glows like a field of small stars."
)
TOPICS = ["a lighthouse keeper", "a robot learning to paint", "a dragon who is afraid of fire", "a lost submarine",
          "a baker on Mars", "a chess champion who is a fox", "a haunted library", "a train that never stops",
          "a village under a glacier", "a clockmaker's apprentice", "a pirate who cannot swim", "a forgotten garden",
          "a violin that plays by itself", "a traveling circus on the moon", "a detective cat", "a mountain that sings",
          "a message in a bottle", "a ghost who loves cooking", "a desert city built of glass", "a boy and his kite",
          "a time-traveling postman", "a whale who writes poems", "a cursed map", "a snowman who wants summer",
          "a night market in the clouds", "a sailor and the last storm", "an orchestra of insects", "a door in the forest",
          "a spaceship full of plants", "a thief who steals sounds", "a village of talking stones", "a girl who collects shadows"]

def prompt_for(workload, i):
    if workload == "copy":
        return (f"Document {i}. Repeat the following passage exactly, word for word.\n\nPassage:\n{PASSAGE}\n\n"
                f"Exact repetition of the passage:\n")
    return f"Write a long, detailed story about {TOPICS[i % len(TOPICS)]}.\n\nStory:\n"

def spec_counters():
    """Sum the spec-decode counters from /metrics (empty dict if speculation is off)."""
    out = {}
    try:
        text = requests.get(f"{URL}/metrics", timeout=10).text
    except Exception:
        return out
    for line in text.splitlines():
        if line.startswith("vllm:spec_decode") and not line.startswith("#"):
            m = re.match(r"(vllm:spec_decode\w+?)(?:\{.*\})?\s+([0-9.eE+-]+)$", line)
            if m and m.group(1).endswith("_total"):
                out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out

def one_request(args):
    i, workload, max_tokens = args
    body = {"model": MODEL, "prompt": prompt_for(workload, i), "max_tokens": max_tokens, "temperature": 0,
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter(); first = None; ntok = 0; text = []
    with requests.post(f"{URL}/v1/completions", json=body, stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "): continue
            data = line[6:]
            if data == b"[DONE]": break
            j = json.loads(data)
            if j.get("choices") and j["choices"][0].get("text"):
                if first is None: first = time.perf_counter()
                text.append(j["choices"][0]["text"])
            if j.get("usage"): ntok = j["usage"]["completion_tokens"]
    t1 = time.perf_counter()
    first = first or t1
    tpot = (t1 - first) / (ntok - 1) * 1000 if ntok > 1 else float("nan")
    return {"ttft_ms": (first - t0) * 1000, "tpot_ms": tpot, "tokens": ntok, "latency_s": t1 - t0}

def pct(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]

def main():
    label, workload, conc = sys.argv[1], sys.argv[2], int(sys.argv[3])
    n = int(sys.argv[4]) if len(sys.argv) > 4 else (20 if conc == 1 else 96)
    max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    assert workload in ("copy", "free")
    one_request((0, workload, 8))                       # warm-up
    before = spec_counters()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(conc) as ex:
        res = list(ex.map(one_request, [(i, workload, max_tokens) for i in range(n)]))
    wall = time.perf_counter() - t0
    after = spec_counters()
    total = sum(r["tokens"] for r in res)
    tp = [r["tpot_ms"] for r in res]; tt = [r["ttft_ms"] for r in res]
    summ = {"label": label, "workload": workload, "concurrency": conc, "requests": n, "max_tokens": max_tokens,
            "mean_tpot_ms": sum(tp) / len(tp), "p99_tpot_ms": pct(tp, 99), "mean_ttft_ms": sum(tt) / len(tt),
            "p99_ttft_ms": pct(tt, 99), "throughput_tok_s": total / wall, "wall_s": wall, "total_tokens": total}
    d = {k: after.get(k, 0) - before.get(k, 0) for k in after}
    drafts = d.get("vllm:spec_decode_num_drafts_total", 0); dtoks = d.get("vllm:spec_decode_num_draft_tokens_total", 0)
    acc = d.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    summ["spec_counters_delta"] = d
    if drafts:
        summ["acceptance_rate"] = acc / dtoks if dtoks else None
        summ["mean_acceptance_length"] = 1 + acc / drafts
    print(json.dumps(summ, indent=2))
    os.makedirs("results", exist_ok=True)
    with open(f"results/{label}_{workload}_c{conc}.json", "w") as f:
        json.dump({"summary": summ, "requests": res}, f, indent=1)

if __name__ == "__main__":
    main()
