"""Day 14: will a model fit on N GPUs with tensor parallelism (TP), and how much KV cache is left?

    python3 memory_plan.py --model qwen3-8b --gpu h100-80 --tp 2 --context 8192

Standard library only. All shapes and sizes come from inputs.md (config.json files and checkpoint index files read on
2026-11-23); GPU memory sizes are NVIDIA's quoted capacities. ASSUMPTIONS (not measurements) are marked ASSUMPTION.
"""
import argparse
import sys

GIB = 1024 ** 3

# name: layers, hidden, kv_heads, head_dim, attn_heads, vocab, checkpoint bytes (safetensors total_size / file size)
MODELS = {
    "qwen3-0.6b": dict(layers=28, hidden=1024, attn_heads=16, kv_heads=8, head_dim=128, vocab=151_936,
                       checkpoint_bytes=1_503_300_328),
    "qwen3-8b": dict(layers=36, hidden=4096, attn_heads=32, kv_heads=8, head_dim=128, vocab=151_936,
                     checkpoint_bytes=16_381_470_720),
    "qwen3-32b": dict(layers=64, hidden=5120, attn_heads=64, kv_heads=8, head_dim=128, vocab=151_936,
                      checkpoint_bytes=65_524_246_528),
    "qwen2.5-72b": dict(layers=80, hidden=8192, attn_heads=64, kv_heads=8, head_dim=128, vocab=152_064,
                        checkpoint_bytes=145_412_407_296),  # head_dim = hidden / attn_heads (not in config)
}

# memory in GiB: ASSUMPTION that NVIDIA's "24 GB / 80 GB / 141 GB" are GiB (the usable amount is a little lower,
# e.g. vLLM saw 23.52 GiB on the 4090; override with --gpu-mem-gib). bandwidth in bytes/s from NVIDIA pages.
GPUS = {
    "rtx4090": dict(mem_gib=24, bandwidth=1008e9),
    "a100-80": dict(mem_gib=80, bandwidth=2039e9),   # caveat in inputs.md: variant not confirmed
    "h100-80": dict(mem_gib=80, bandwidth=3.35e12),  # H100 SXM
    "h200": dict(mem_gib=141, bandwidth=4.8e12),     # H200 SXM
}

DTYPE_BYTES = {"bf16": 2, "fp8": 1}


def plan(model, gpu, tp, dtype="bf16", util=0.9, context=8192, gpu_mem_gib=None,
         activation_gib=2.0, overhead_gib=0.06, min_seqs=4):
    """Return a dict of per-GPU numbers and a verdict. ASSUMPTION defaults: activation 2.0 GiB, overhead 0.06 GiB."""
    m, g = MODELS[model], GPUS[gpu]
    if m["attn_heads"] % tp != 0:
        raise ValueError(f"TP={tp} must divide the {m['attn_heads']} attention heads")
    if not (m["kv_heads"] % tp == 0 or tp % m["kv_heads"] == 0):
        raise ValueError(f"TP={tp} must divide, or be a multiple of, the {m['kv_heads']} KV heads")
    bytes_per = DTYPE_BYTES[dtype]
    mem_gib = gpu_mem_gib if gpu_mem_gib is not None else g["mem_gib"]
    budget = util * mem_gib * GIB

    # ASSUMPTION: weights split evenly across GPUs (norm weights are replicated but tiny); fp8 halves every weight
    # (really the embedding table stays BF16, so this is a small underestimate).
    scale = bytes_per / 2
    weights_gpu = m["checkpoint_bytes"] * scale / tp
    embed_table = m["vocab"] * m["hidden"] * 2 * scale
    weights_read_gpu = (m["checkpoint_bytes"] * scale - embed_table) / tp   # decode reads all but the embedding table
    # KV: heads are split across GPUs, but a head is never split: at least 1 per GPU (replicate-heads rule).
    kv_heads_gpu = max(1, m["kv_heads"] / tp)
    kv_per_token_gpu = 2 * m["layers"] * kv_heads_gpu * m["head_dim"] * bytes_per

    fixed = weights_gpu + (activation_gib + overhead_gib) * GIB
    free = budget - fixed
    if free <= 0:
        tokens = seqs = 0
        verdict = "doesn't fit"
    else:
        tokens = int(free // kv_per_token_gpu)   # each GPU stores its share of every token's KV, so tokens do not add up
        seqs = tokens // context
        verdict = "fits" if seqs >= min_seqs else f"fits with too little KV (< {min_seqs} sequences of {context})"
    return dict(model=model, gpu=gpu, tp=tp, dtype=dtype, util=util, context=context, budget_gib=budget / GIB,
                weights_gpu=weights_gpu, weights_read_gpu=weights_read_gpu, kv_per_token_gpu=kv_per_token_gpu,
                kv_heads_gpu=kv_heads_gpu, free_gib=free / GIB, kv_tokens=tokens, kv_seqs=seqs, verdict=verdict)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--gpu", required=True, choices=GPUS)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--dtype", default="bf16", choices=DTYPE_BYTES)
    p.add_argument("--util", type=float, default=0.9, help="gpu_memory_utilization")
    p.add_argument("--context", type=int, default=8192, help="tokens per sequence")
    p.add_argument("--gpu-mem-gib", type=float, default=None, help="override usable GPU memory (GiB)")
    p.add_argument("--activation-gib", type=float, default=2.0, help="ASSUMPTION: peak activation memory per GPU")
    p.add_argument("--overhead-gib", type=float, default=0.06, help="ASSUMPTION: non-weight, non-activation overhead")
    a = p.parse_args(argv)
    try:
        r = plan(a.model, a.gpu, a.tp, a.dtype, a.util, a.context, a.gpu_mem_gib, a.activation_gib, a.overhead_gib)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"{r['model']} on {r['tp']} x {r['gpu']}, {r['dtype']}, util {r['util']}, context {r['context']:,}")
    print(f"  budget per GPU:            {r['budget_gib']:.2f} GiB")
    print(f"  weights per GPU:           {r['weights_gpu']:,.0f} bytes ({r['weights_gpu']/GIB:.2f} GiB)")
    print(f"  weights read per step/GPU: {r['weights_read_gpu']:,.0f} bytes (checkpoint minus embedding table, split by TP)")
    print(f"  KV heads per GPU:          {r['kv_heads_gpu']:g}")
    print(f"  KV bytes/token per GPU:    {r['kv_per_token_gpu']:,.0f}")
    print(f"  free for KV per GPU:       {r['free_gib']:.2f} GiB (after weights, activation, overhead)")
    print(f"  KV capacity:               {r['kv_tokens']:,} tokens = {r['kv_seqs']:,} sequences of {r['context']:,}")
    print(f"  verdict:                   {r['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
