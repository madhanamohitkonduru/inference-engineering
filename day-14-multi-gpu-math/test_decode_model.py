import unittest

import decode_model as dm
import memory_plan as mp


class Decode(unittest.TestCase):
    def test_tp1_is_pure_memory_time_no_communication(self):
        mt, ct = dm.step_time("qwen3-8b", "h100-80", 1, "nvlink", context=1024, batch=1)
        r = mp.plan("qwen3-8b", "h100-80", 1)
        self.assertEqual(ct, 0.0)
        self.assertAlmostEqual(mt, (r["weights_read_gpu"] + 1024 * r["kv_per_token_gpu"]) / 3.35e12, places=12)

    def test_allreduce_count_is_two_per_layer(self):
        _, ct = dm.step_time("qwen3-8b", "h100-80", 2, "nvlink", latency=1e-6, link_bw=1e30)   # bandwidth term ~ 0
        self.assertAlmostEqual(ct, 2 * 36 * 1e-6, places=12)

    def test_more_latency_is_slower_and_more_link_bandwidth_is_not_slower(self):
        a = dm.tps("qwen3-8b", "h100-80", 2, "nvlink", latency=5e-6)
        b = dm.tps("qwen3-8b", "h100-80", 2, "nvlink", latency=50e-6)
        self.assertGreater(a, b)
        c = dm.tps("qwen3-8b", "h100-80", 2, "pcie4", link_bw=10e9)
        d = dm.tps("qwen3-8b", "h100-80", 2, "pcie4", link_bw=100e9)
        self.assertGreaterEqual(d, c)

    def test_memory_time_halves_with_tp2(self):
        m1, _ = dm.step_time("qwen3-8b", "rtx4090", 1, "pcie4", context=1)
        m2, _ = dm.step_time("qwen3-8b", "rtx4090", 2, "pcie4", context=1)
        self.assertAlmostEqual(m1 / m2, 2.0, places=3)


if __name__ == "__main__":
    unittest.main()
