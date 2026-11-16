"""全局 Smax 排序与分级阈值。

算法（冻结）：
  1. 汇总 35 个 Tensor 的全部权重；
  2. 按 Smax 全局降序；
  3. 前 5% → High，接下来 15% → Medium，其余 80% → Low。

实现：用 log10(Smax) 的精细直方图求分位阈值（单遍扫描，内存 O(1)）。
阈值确定后，逐 Tensor 落盘 level_map（uint8: 0/1/2）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..constants import LEVEL_CODES, LEVEL_RATIOS
from .source import ImportanceSource

N_BINS = 20000
CHUNK = 3_000_000


class GlobalRanker:
    def __init__(self, source: ImportanceSource, n_bins: int = N_BINS):
        self.source = source
        self.n_bins = n_bins

    # ---------------- 阈值 ----------------
    def _scan_range(self) -> tuple[float, float]:
        lo, hi = np.inf, -np.inf
        for name in self.source.tensors:
            arr = self.source.smax_log10(name)
            n = arr.shape[0]
            for s in range(0, n, CHUNK):
                block = np.asarray(arr[s:min(s + CHUNK, n)], dtype=np.float64)
                if block.size:
                    lo = min(lo, float(block.min()))
                    hi = max(hi, float(block.max()))
            del arr
        return lo, hi

    def _histogram(self, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
        edges = np.linspace(lo - 1e-6, hi + 1e-6, self.n_bins + 1)
        counts = np.zeros(self.n_bins, dtype=np.int64)
        for name in self.source.tensors:
            arr = self.source.smax_log10(name)
            n = arr.shape[0]
            for s in range(0, n, CHUNK):
                block = np.asarray(arr[s:min(s + CHUNK, n)], dtype=np.float64)
                if block.size:
                    idx = np.clip(np.searchsorted(edges, block, side="right") - 1,
                                  0, self.n_bins - 1)
                    counts += np.bincount(idx, minlength=self.n_bins)
                del block
            del arr
        return counts, edges

    def thresholds(self) -> dict:
        lo, hi = self._scan_range()
        counts, edges = self._histogram(lo, hi)
        total = int(counts.sum())
        cumulative = np.cumsum(counts[::-1])          # 由高到低累计
        out = {"total_weights": total, "log10_range": [lo, hi],
               "ratios": dict(LEVEL_RATIOS)}
        k_high = int(round(total * LEVEL_RATIOS["high"]))
        k_high_med = int(round(total * (LEVEL_RATIOS["high"] +
                                        LEVEL_RATIOS["medium"])))
        for label, k in (("high", k_high), ("medium_boundary", k_high_med)):
            idx = int(np.searchsorted(cumulative, k))
            bin_index = self.n_bins - 1 - idx
            out[f"{label}_threshold_smax"] = float(10.0 ** edges[bin_index])
            out[f"{label}_threshold_log10"] = float(edges[bin_index])
            out[f"{label}_target_count"] = k
        return out

    # ---------------- 落盘 ----------------
    def write_level_maps(self, thresholds: dict, out_dir: str | Path) -> dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        thr_high = thresholds["high_threshold_smax"]
        thr_med = thresholds["medium_boundary_threshold_smax"]
        summary = {}
        for name, spec in self.source.tensors.items():
            arr = self.source.smax_log10(name)
            target = out_dir / f"L{spec.block_id}_{spec.module}"
            target.mkdir(parents=True, exist_ok=True)
            level_map = np.lib.format.open_memmap(
                target / "level_map.npy", mode="w+", dtype=np.uint8,
                shape=(spec.n_weights,))
            n_high = n_med = 0
            for s in range(0, spec.n_weights, CHUNK):
                e = min(s + CHUNK, spec.n_weights)
                block = np.asarray(arr[s:e], dtype=np.float64)
                smax = np.power(10.0, block)
                level = np.zeros(e - s, dtype=np.uint8)
                level[smax >= thr_med] = LEVEL_CODES["medium"]
                level[smax >= thr_high] = LEVEL_CODES["high"]
                level_map[s:e] = level
                n_high += int((level == LEVEL_CODES["high"]).sum())
                n_med += int((level == LEVEL_CODES["medium"]).sum())
                del block, smax, level
            level_map.flush()
            del level_map, arr
            summary[name] = {"n_weights": spec.n_weights, "high": n_high,
                             "medium": n_med, "low": spec.n_weights - n_high - n_med}
        (out_dir / "allocation_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
