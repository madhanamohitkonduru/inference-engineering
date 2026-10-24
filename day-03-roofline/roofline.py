"""Day 3: KV-cache capacity math and a decode roofline for Qwen3-0.6B on an RTX 4090.

Reads measured values from the Day 1 benchmark results in this repo, so the
predicted-vs-measured table is reproducible:  python3 roofline.py
"""
import json
from pathlib import Path

GIB = 1024 ** 3

# ---- Inputs --------------------------------------------------------------
# Model (Qwen/Qwen3-0.6B config.json, checked 2026-10-24)
LAYERS, KV_HEADS, HEAD_DIM = 28, 8, 128
HIDDEN, INTERMEDIATE, N_HEADS, VOCAB = 1024, 3072, 16, 151_936
BYTES_PER_ELEM = 2                      # BF16
WEIGHT_BYTES_CHECKPOINT = 1.40 * GIB    # "Checkpoint size" line in the vLLM log (initial Day 3 estimate)

# Bytes actually read per decode step: the layer matmul weights plus the lm_head matrix (tied to the embedding,
# read in full to compute logits). The embedding lookup reads only one row per token, so the rest of the
# checkpoint file is not read each step.
_qkv = HIDDEN * (N_HEADS * HEAD_DIM + 2 * KV_HEADS * HEAD_DIM)
_o = N_HEADS * HEAD_DIM * HIDDEN
_gate_up = HIDDEN * 2 * INTERMEDIATE
_down = INTERMEDIATE * HIDDEN
LAYER_WEIGHT_BYTES = LAYERS * (_qkv + _o + _gate_up + _down) * BYTES_PER_ELEM
LM_HEAD_BYTES = VOCAB * HIDDEN * BYTES_PER_ELEM
WEIGHT_BYTES = LAYER_WEIGHT_BYTES + LM_HEAD_BYTES

# vLLM startup log (Day 1, gpu_memory_utilization=0.85)
GPU_TOTAL_GIB = 23.52
UTIL = 0.85
CONSUMED_GIB = 1.46                     # weights + non-torch
ACTIVATION_GIB = 0.61                   # peak activation (profiled at startup)
LOG_KV_TOKENS = 167_728
LOG_KV_GIB = 17.92

# Hardware: RTX 4090, 21 Gbps x 384-bit bus / 8 bits per byte
BANDWIDTH = 21e9 * 384 / 8              # = 1.008e12 bytes/s

# Benchmark shape: 512 input + 128 output tokens; average context during decode
AVG_CONTEXT = 512 + 128 // 2            # = 576 tokens

# ---- KV cache math ---------------------------------------------------------
kv_per_token = 2 * LAYERS * KV_HEADS * HEAD_DIM * BYTES_PER_ELEM
print(f"KV bytes/token: {kv_per_token:,} ({kv_per_token/1024:.0f} KiB)")
print(f"KV bytes for 1,000 tokens: {kv_per_token*1000:,} ({kv_per_token*1000/2**20:.1f} MiB)")

budget = UTIL * GPU_TOTAL_GIB
naive = (budget - 1.40) * GIB / kv_per_token
corrected_gib = budget - CONSUMED_GIB - ACTIVATION_GIB
corrected = corrected_gib * GIB / kv_per_token
print(f"\nBudget: {budget:.3f} GiB")
print(f"Naive (budget - weights):                 {naive:,.0f} tokens ({(naive/LOG_KV_TOKENS-1)*100:+.1f}% vs log)")
print(f"Corrected (budget - consumed - activation): {corrected_gib:.2f} GiB = {corrected:,.0f} tokens ({(corrected/LOG_KV_TOKENS-1)*100:+.2f}% vs log)")
print(f"vLLM log: {LOG_KV_TOKENS:,} tokens ({LOG_KV_GIB} GiB)")
print(f"1,000-token sequences that fit: {LOG_KV_TOKENS//1000}; "
      f"if each reserved 4,096 tokens (max_model_len): {LOG_KV_TOKENS//4096}")

# ---- Decode roofline -------------------------------------------------------
# Per step, read all weights once plus each sequence's KV. One token per sequence per step.
res = Path(__file__).resolve().parent.parent / "day-01-vllm-baseline" / "results"
kv_read = AVG_CONTEXT * kv_per_token
print(f"\nBandwidth: {BANDWIDTH/1e9:.0f} GB/s; KV read per sequence per step {kv_read/1e9:.3f} GB (context {AVG_CONTEXT})")
print(f"Weights read per step: layers {LAYER_WEIGHT_BYTES/1e6:.1f} MB + lm_head {LM_HEAD_BYTES/1e6:.1f} MB = {WEIGHT_BYTES/1e9:.3f} GB "
      f"(initial estimate used the {WEIGHT_BYTES_CHECKPOINT/1e9:.3f} GB checkpoint size)")

def predict(c, weights):
    return c / ((weights + c * kv_read) / BANDWIDTH)

print("\n| Concurrency | GB/step | Predicted tok/s | Measured output tok/s | Gap | Measured decode-only tok/s | Gap |")
print("|---|---|---|---|---|---|---|")
init = []
for c in (1, 8, 32):
    d = json.load(open(res / f"c{c}.json"))
    mo, md = d["output_throughput"], c * 1000 / d["mean_tpot_ms"]
    pred = predict(c, WEIGHT_BYTES)
    init.append((c, predict(c, WEIGHT_BYTES_CHECKPOINT), md))
    print(f"| {c} | {(WEIGHT_BYTES + c*kv_read)/1e9:.3f} | {pred:,.0f} | {mo:,.0f} | {(pred-mo)/pred*100:.0f}% | {md:,.0f} | {(pred-md)/pred*100:.0f}% |")

print("\nInitial (superseded) estimate using checkpoint size as weight bytes, decode-only gap:")
for c, pred, md in init:
    print(f"  c={c}: predicted {pred:,.0f} tok/s, gap {(pred-md)/pred*100:.0f}%")
