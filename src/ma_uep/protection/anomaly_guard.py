"""Anomaly Guard：逐 Tensor 最大幅值检测与置零。

规则（冻结）：A(w) = NonFinite(w) OR (|w| > T_k) → 置零。
T_k 来自干净模型该 Tensor 的最大 |w|，**不使用全局统一阈值**。

实现为分块流式处理：单 Tensor 可达 1376 万权重，一次性转换为 float32
会申请 55 MB 以上的临时数组，在本机（可用提交内存约 800 MB）会失败。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..constants import GUARD_RULE
from ..paths import PROJECT_ROOT

CHUNK = 1_000_000


def _to_float(raw_u16: np.ndarray) -> np.ndarray:
    return torch.from_numpy(raw_u16.astype(np.int16)) \
        .view(torch.bfloat16).float().numpy()


class AnomalyGuard:
    def __init__(self, thresholds: dict[str, float], chunk: int = CHUNK):
        self.thresholds = dict(thresholds)
        self.chunk = chunk

    # ---------------- 构建 ----------------
    @classmethod
    def from_file(cls, path: str | Path, repo_root: Path | None = None):
        p = Path(path)
        if not p.is_absolute():
            root = repo_root or PROJECT_ROOT
            p = root / p
        payload = json.loads(p.read_text(encoding="utf-8"))
        return cls(payload["thresholds"])

    @classmethod
    def compute(cls, model_manager, specs) -> "AnomalyGuard":
        thresholds = {}
        for spec in specs:
            raw = model_manager.read_raw(spec)
            max_abs = 0.0
            for start in range(0, len(raw), CHUNK):
                values = _to_float(raw[start:start + CHUNK])
                max_abs = max(max_abs, float(np.abs(values).max()))
            thresholds[spec.name] = max_abs
        return cls(thresholds)

    def threshold(self, tensor_name: str) -> float:
        if tensor_name not in self.thresholds:
            raise KeyError(f"缺少阈值: {tensor_name}")
        return self.thresholds[tensor_name]

    # ---------------- 检测 ----------------
    def detect(self, raw_u16: np.ndarray, tensor_name: str) -> dict:
        threshold = self.threshold(tensor_name)
        n_nonfinite = n_over = 0
        max_abs = 0.0
        position_chunks = []
        for start in range(0, len(raw_u16), self.chunk):
            block = raw_u16[start:start + self.chunk]
            values = _to_float(block)
            finite = np.isfinite(values)
            magnitude = np.abs(values)
            over = magnitude > threshold
            anomaly = (~finite) | over
            n_nonfinite += int((~finite).sum())
            n_over += int(over.sum())
            if finite.any():
                max_abs = max(max_abs, float(magnitude[finite].max()))
            if anomaly.any():
                position_chunks.append(np.flatnonzero(anomaly) + start)
            del values, finite, magnitude, over, anomaly
        positions = (np.concatenate(position_chunks) if position_chunks
                     else np.empty(0, dtype=np.int64))
        return {
            "tensor": tensor_name,
            "n_weights": int(len(raw_u16)),
            "threshold": threshold,
            "n_nonfinite": n_nonfinite,
            "n_over_threshold": n_over,
            "n_anomaly": int(positions.size),
            "max_abs": max_abs,
            "positions": positions,
        }

    # ---------------- 置零 ----------------
    def apply(self, raw_u16: np.ndarray, tensor_name: str) -> tuple[np.ndarray, dict]:
        stats = self.detect(raw_u16, tensor_name)
        cleaned = raw_u16.copy()
        cleaned[stats.pop("positions")] = 0
        return cleaned, stats

    def apply_by_level(self, raw_u16: np.ndarray, tensor_name: str,
                       levels: np.ndarray) -> tuple[np.ndarray, dict]:
        stats = self.detect(raw_u16, tensor_name)
        positions = stats.pop("positions")
        cleaned = raw_u16          # 就地置零：调用方已持有独立副本
        cleaned[positions] = 0
        per_level = {}
        for code, level in ((0, "low"), (1, "medium"), (2, "high")):
            count = 0
            for start in range(0, len(levels), self.chunk):
                block_levels = levels[start:start + self.chunk]
                mask = block_levels == code
                if not mask.any():
                    continue
                window = positions[(positions >= start) &
                                   (positions < start + self.chunk)]
                count += int(mask[window - start].sum())
            per_level[level] = count
        stats["per_level"] = per_level
        return cleaned, stats

    def describe(self) -> dict:
        return {"rule": GUARD_RULE, "n_tensors": len(self.thresholds),
                "thresholds": self.thresholds}
