"""Compare saved outputs against a reference.  Usage: python quality_compare.py <reference_label> <label> [<label> ...]
For each label: how many outputs are word-for-word identical to the reference, and where the others first diverge
(position in tokens, found by tokenizing the shared text). Prints every differing pair (shortened)."""
import json, os, statistics, sys
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
load = lambda l: json.load(open(f"outputs_{l}.json"))
ref_label, others = sys.argv[1], sys.argv[2:]
ref = load(ref_label)
for label in others:
    cur = load(label)
    same, positions, diffs = 0, [], []
    for i, (a, b) in enumerate(zip(ref, cur)):
        if a["text"] == b["text"]:
            same += 1
            continue
        common = os.path.commonprefix([a["text"], b["text"]])
        pos = len(tok.encode(common)) if common else 0
        positions.append(pos)
        diffs.append((i, pos, common, a["text"][len(common):], b["text"][len(common):]))
    print(f"\n===== {label} vs {ref_label}: {same} of {len(ref)} identical =====")
    if positions:
        print(f"first divergence at token position: min {min(positions)}, median {statistics.median(positions):.0f}, max {max(positions)}; all: {sorted(positions)}")
    for i, pos, common, ra, ca in diffs:
        print(f"\n[{i}] prompt {ref[i]['prompt']!r}  diverges at ~token {pos}\n   ...{common[-40:]!r}\n   {ref_label}: {ra[:90]!r}\n   {label}: {ca[:90]!r}")
