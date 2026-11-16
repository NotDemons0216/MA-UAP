"""UEP 分配与开销统计测试（标准库 unittest）。"""

import unittest

import numpy as np

from ma_uep.constants import BCH_HIGH, BCH_MEDIUM
from ma_uep.uep.allocator import allocation_stats, validate


class TestAllocator(unittest.TestCase):
    def test_stats_math(self):
        total = 1_000_000
        counts = {"high": 50_000, "medium": 150_000, "low": 800_000}
        stats = allocation_stats(counts, total)
        self.assertEqual(stats["parity_bits"],
                         BCH_HIGH["parity_bits"] * 50_000 +
                         BCH_MEDIUM["parity_bits"] * 150_000)
        self.assertEqual(stats["data_bits"], 16 * total)
        self.assertEqual(stats["total_bits"],
                         stats["data_bits"] + stats["parity_bits"])
        self.assertAlmostEqual(stats["code_rate"],
                               stats["data_bits"] / stats["total_bits"], places=12)
        self.assertAlmostEqual(stats["high_frac"], 0.05, places=12)

    def test_rejects_bad_total(self):
        with self.assertRaises(ValueError):
            allocation_stats({"high": 1, "medium": 1, "low": 1}, 10)

    def test_frozen_code_rate_matches_history(self):
        total = 233_963_520
        counts = {"high": 11_797_359, "medium": 35_150_400,
                  "low": 233_963_520 - 11_797_359 - 35_150_400}
        stats = allocation_stats(counts, total)
        self.assertAlmostEqual(stats["code_rate"], 0.890954, places=5)
        self.assertEqual(stats["parity_bits"], 458_163_585)

    def test_validate_level_codes(self):
        maps = {"L0.q_proj": np.array([0, 1, 2, 0], dtype=np.uint8)}
        self.assertEqual(validate(maps)["total_weights"], 4)
        with self.assertRaises(ValueError):
            validate({"L0.q_proj": np.array([0, 3], dtype=np.uint8)})


if __name__ == "__main__":
    unittest.main()
