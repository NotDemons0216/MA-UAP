"""Importance 生成链路的算法测试（不需要模型 / GPU）。"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ma_uep.constants import BF16_BITS, EXCLUDED_BF16_BITS
from ma_uep.importance.ranker import GlobalRanker
from ma_uep.importance.scan import RISK_BITS, scan_tensor
from ma_uep.importance.source import ImportanceSource

RISK = tuple(b for b in range(BF16_BITS) if b not in EXCLUDED_BF16_BITS)


class _Spec:
    def __init__(self, n, K, name="L0.q_proj"):
        self.name = name
        self.block_id = 0
        self.module = "q_proj"
        self.n_weights = n
        self.shape = (n // K, K)


class _StubManager:
    """只需 read_raw，不加载模型。"""

    def __init__(self, raw):
        self._raw = raw

    def read_raw(self, spec):
        return self._raw


def _to_float(raw):
    return torch.from_numpy(raw.astype(np.int16)).view(torch.bfloat16).double().numpy()


class TestScan(unittest.TestCase):
    def test_smax_matches_bruteforce(self):
        rng = np.random.default_rng(2027)
        n, K = 600, 12
        values = (rng.normal(size=n) * 0.05).astype(np.float32)
        tensor = torch.from_numpy(values).to(torch.bfloat16)
        raw = tensor.view(torch.int16).numpy().view(np.uint16)
        F = rng.uniform(0.1, 2.0, size=K)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            scan_tensor(_StubManager(raw), _Spec(n, K), F, out, chunk=97)
            stored = np.load(out / "I_max_log10.npy")

        # 暴力重算
        expected = np.empty(n)
        for i in range(n):
            base = _to_float(np.array([raw[i]]))[0]
            best = 0.0
            for b in RISK:
                flipped = np.array([raw[i]], dtype=np.uint16) ^ np.uint16(1 << b)
                delta = abs(_to_float(flipped)[0] - base)
                best = max(best, F[i % K] * delta)
            expected[i] = best
        with np.errstate(divide="ignore"):
            expected_log = np.log10(expected, where=expected > 0,
                                    out=np.full(n, -np.inf))
        np.testing.assert_allclose(stored, expected_log, rtol=1e-5, atol=1e-6)

    def test_max_abs_reported(self):
        rng = np.random.default_rng(7)
        n, K = 200, 4
        tensor = torch.from_numpy((rng.normal(size=n) * 0.3).astype(np.float32)) \
            .to(torch.bfloat16)
        raw = tensor.view(torch.int16).numpy().view(np.uint16)
        with tempfile.TemporaryDirectory() as tmp:
            meta = scan_tensor(_StubManager(raw), _Spec(n, K),
                               np.ones(K), Path(tmp), chunk=64)
        self.assertAlmostEqual(meta["max_abs"],
                               float(abs(tensor.float()).max()), places=4)


class TestRanker(unittest.TestCase):
    def test_thresholds_split_as_configured(self):
        rng = np.random.default_rng(11)
        root = Path(tempfile.mkdtemp())
        names = ["L0.q_proj", "L0.k_proj"]
        sizes = [4000, 6000]
        for name, size in zip(names, sizes):
            block, module = name.split(".")
            folder = root / f"{block}_{module}"
            folder.mkdir(parents=True)
            log10 = rng.normal(size=size).astype(np.float32)
            np.save(folder / "I_max_log10.npy", log10)
            (folder / "meta.json").write_text(json.dumps(
                {"name": name, "shape": [size, 1]}), encoding="utf-8")
        source = ImportanceSource(root, (0,), ("q_proj", "k_proj"))
        ranker = GlobalRanker(source)
        thresholds = ranker.thresholds()
        self.assertEqual(thresholds["total_weights"], 10000)
        counts = ranker.write_level_maps(thresholds, root / "alloc")
        high = sum(c["high"] for c in counts.values())
        medium = sum(c["medium"] for c in counts.values())
        self.assertAlmostEqual(high / 10000, 0.05, delta=0.01)
        self.assertAlmostEqual(medium / 10000, 0.15, delta=0.01)


if __name__ == "__main__":
    unittest.main()
