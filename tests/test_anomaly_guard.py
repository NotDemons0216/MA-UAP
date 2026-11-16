"""Anomaly Guard 测试（标准库 unittest）。"""

import unittest

import numpy as np
import torch

from ma_uep.protection.anomaly_guard import AnomalyGuard


def _raw(values):
    tensor = torch.tensor(values, dtype=torch.float32).to(torch.bfloat16)
    return tensor.view(torch.int16).numpy().view(np.uint16)


def _float(raw):
    return torch.from_numpy(raw.astype(np.int16)).view(torch.bfloat16).float()


class TestAnomalyGuard(unittest.TestCase):
    def setUp(self):
        self.guard = AnomalyGuard({"L0.q_proj": 1.0})

    def test_normal_untouched(self):
        cleaned, stats = self.guard.apply(_raw([0.5, -0.25, 0.0, 0.75]),
                                          "L0.q_proj")
        self.assertEqual(stats["n_anomaly"], 0)
        self.assertTrue(np.array_equal(cleaned, _raw([0.5, -0.25, 0.0, 0.75])))

    def test_threshold_equality_not_zeroed(self):
        _, stats = self.guard.apply(_raw([1.0, -1.0, 0.999]), "L0.q_proj")
        self.assertEqual(stats["n_anomaly"], 0)

    def test_over_threshold_zeroed(self):
        cleaned, stats = self.guard.apply(_raw([1.5, -2.0, 0.5]), "L0.q_proj")
        self.assertEqual(stats["n_anomaly"], 2)
        values = _float(cleaned)
        self.assertEqual(values[0].item(), 0.0)
        self.assertEqual(values[1].item(), 0.0)
        self.assertAlmostEqual(values[2].item(), 0.5, places=3)

    def test_inf_nan_detected(self):
        raw = _raw([float("inf"), float("-inf"), float("nan"), 0.5])
        _, stats = self.guard.apply(raw, "L0.q_proj")
        self.assertEqual(stats["n_nonfinite"], 3)
        self.assertGreaterEqual(stats["n_anomaly"], 3)

    def test_per_level_counts(self):
        raw = _raw([1.5, 0.1, 2.0, 0.2])
        levels = np.array([2, 0, 1, 0], dtype=np.uint8)
        _, stats = self.guard.apply_by_level(raw, "L0.q_proj", levels)
        self.assertEqual(stats["per_level"]["high"], 1)
        self.assertEqual(stats["per_level"]["medium"], 1)
        self.assertEqual(stats["per_level"]["low"], 0)

    def test_missing_threshold(self):
        with self.assertRaises(KeyError):
            self.guard.threshold("L99.nope")


if __name__ == "__main__":
    unittest.main()
