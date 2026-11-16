"""信道注入测试（标准库 unittest）：位置规则、可复现性、与旧实现逐位一致。"""

import unittest

import numpy as np

from ma_uep.channel.injection import (ChannelConfig, allowed_positions,
                                      build_channel)


class TestChannel(unittest.TestCase):
    def test_allowed_positions_excludes_b14(self):
        allowed = allowed_positions(31, 15)
        self.assertFalse(allowed[15 + 14])
        self.assertEqual(int(allowed.sum()), 30)

        allowed_med = allowed_positions(24, 8)
        self.assertFalse(allowed_med[8 + 14])
        self.assertEqual(int(allowed_med.sum()), 23)

        allowed_low = allowed_positions(16, 0)
        self.assertFalse(allowed_low[14])
        self.assertEqual(int(allowed_low.sum()), 15)

    def test_ber_reproducible(self):
        a = build_channel(ChannelConfig(mode="ber", ber=1e-2, seed=123))
        b = build_channel(ChannelConfig(mode="ber", ber=1e-2, seed=123))
        wa, ba, na = a.sample(1000, 31, 15)
        wb, bb, nb = b.sample(1000, 31, 15)
        self.assertEqual(na, nb)
        self.assertTrue(np.array_equal(wa, wb))
        self.assertTrue(np.array_equal(ba, bb))

    def test_ber_zero(self):
        channel = build_channel(ChannelConfig(mode="ber", ber=0.0, seed=1))
        words = np.zeros(100, dtype=np.uint32)
        w, b, n = channel.sample(100, 24, 8)
        self.assertEqual(n, 0)
        channel.apply(words, w, b)
        self.assertTrue((words == 0).all())

    def test_fixed_weight_count(self):
        channel = build_channel(ChannelConfig(mode="fixed", bad_weights=50,
                                              k_bits=3, seed=7))
        channel.begin_tensor(500)
        self.assertEqual(len(channel.bad_weights_all), 50)
        bits = channel.draw_bits(50, 31, 15)
        self.assertEqual(bits.size, 150)

    def test_b14_never_corrupted(self):
        channel = build_channel(ChannelConfig(mode="ber", ber=0.05, seed=11))
        words = np.zeros(20000, dtype=np.uint32)
        w, b, _ = channel.sample(20000, 31, 15)
        channel.apply(words, w, b)
        b14_mask = np.uint32(1 << (15 + 14))
        self.assertEqual(int(((words & b14_mask) != 0).sum()), 0)


class TestMatchesLegacyImplementation(unittest.TestCase):
    """与旧脚本 ECC/simulation/apply_ecc.py 的随机流逐位一致。"""

    def setUp(self):
        import sys
        from pathlib import Path
        repo = Path(__file__).resolve().parents[2]
        old_dir = repo / "ECC" / "simulation"
        if not old_dir.exists():
            self.skipTest("旧实现目录不存在")
        sys.path.insert(0, str(old_dir))
        try:
            import importlib
            self.legacy = importlib.import_module("apply_ecc")
        except Exception as exc:                      # noqa: BLE001
            self.skipTest(f"无法导入旧实现: {exc}")

    def test_ber_stream_identical(self):
        from ma_uep.channel.injection import BerChannel
        n_words, n_bits, offset = 5000, 31, 15
        words = np.zeros(n_words, dtype=np.uint32)
        rng = np.random.default_rng(42)
        old_out, old_n = self.legacy.inject_errors(
            words, n_bits, 1e-3, rng,
            self.legacy.channel_allowed_positions(n_bits, offset))

        channel = BerChannel(ChannelConfig(mode="ber", ber=1e-3, seed=42))
        w, b, new_n = channel.sample(n_words, n_bits, offset)
        new_out = words.copy()
        channel.apply(new_out, w, b)

        self.assertEqual(old_n, new_n)
        self.assertTrue(np.array_equal(old_out, new_out),
                        "新信道与旧实现的错误图样不一致")


if __name__ == "__main__":
    unittest.main()
