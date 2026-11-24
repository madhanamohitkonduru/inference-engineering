"""Tests for memory_plan.py. First test: reproduce the Day 3 corrected estimate (no GPU needed)."""
import unittest

import memory_plan as mp

DAY3_LOG_TOKENS = 167_728        # vLLM startup log, Day 1
DAY3_CORRECTED_TOKENS = 167_791  # Day 3 corrected estimate
DAY3_READ_BYTES = 1.192e9        # Day 3 roofline.py: layer matmuls + lm_head


def day3(**kw):
    # Day 3 inputs: usable memory 23.52 GiB, utilization 0.85, activation 0.61 GiB, overhead 0.06 GiB.
    args = dict(model="qwen3-0.6b", gpu="rtx4090", tp=1, util=0.85, context=1000, gpu_mem_gib=23.52,
                activation_gib=0.61, overhead_gib=0.06)
    args.update(kw)
    return mp.plan(**args)


class Day3(unittest.TestCase):
    def test_kv_bytes_per_token(self):
        self.assertEqual(day3()["kv_per_token_gpu"], 114_688)

    def test_reproduces_day3_within_0_1_percent_of_vllm(self):
        r = day3()
        self.assertLess(abs(r["kv_tokens"] / DAY3_LOG_TOKENS - 1), 0.001)
        self.assertLess(abs(r["kv_tokens"] / DAY3_CORRECTED_TOKENS - 1), 0.001)

    def test_weights_read_per_step_matches_day3(self):
        self.assertLess(abs(day3()["weights_read_gpu"] / DAY3_READ_BYTES - 1), 0.001)


class Rules(unittest.TestCase):
    def test_replicate_heads_rule(self):
        # 8 KV heads: TP=2 -> 4 per GPU; TP=8 -> 1; TP=16 -> still 1 (copies), so KV per token per GPU stops shrinking
        k = {tp: mp.plan("qwen3-8b", "h100-80", tp)["kv_per_token_gpu"] for tp in (1, 2, 8)}
        self.assertEqual(k[1] / 8, k[8])
        self.assertEqual(k[1] / 2, k[2])
        m = dict(mp.MODELS["qwen3-8b"]); m["attn_heads"] = 64
        mp.MODELS["_t"] = m
        try:
            self.assertEqual(mp.plan("_t", "h100-80", 16)["kv_per_token_gpu"], k[8])
        finally:
            del mp.MODELS["_t"]

    def test_bad_tp_raises(self):
        with self.assertRaises(ValueError):
            mp.plan("qwen3-8b", "h100-80", 3)

    def test_verdicts(self):
        v = lambda m, g, t: mp.plan(m, g, t)["verdict"]
        self.assertEqual(v("qwen3-32b", "rtx4090", 1), "doesn't fit")
        self.assertEqual(v("qwen3-8b", "h100-80", 1), "fits")
        self.assertTrue(v("qwen3-8b", "rtx4090", 1).startswith("fits with too little KV"))
        self.assertTrue(v("qwen2.5-72b", "h100-80", 2).startswith("fits with too little KV"))

    def test_kv_more_than_doubles_when_weights_dominate(self):
        a = mp.plan("qwen3-32b", "a100-80", 1, util=0.9)["kv_tokens"]
        b = mp.plan("qwen3-32b", "a100-80", 2, util=0.9)["kv_tokens"]
        self.assertGreater(b, 2 * a)


if __name__ == "__main__":
    unittest.main()
