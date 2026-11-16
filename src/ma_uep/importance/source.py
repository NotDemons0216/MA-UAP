"""Importance Analyzer：读取已有的权重级 Smax 并做全局排序。

设计要点：
  - **不重新计算** Bit-level 风险（16 GB，成本高）；
  - 只读取 task1 产出的 `I_max_log10.npy`（log10 Smax）；
  - 通过 memmap 流式扫描，避免把 2.34 亿个浮点一次性读入内存。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..constants import BLOCKS, MODULES, COVERAGE_NOTE


@dataclass(frozen=True)
class TensorSpec:
    block_id: int
    module: str
    n_weights: int
    shape: tuple[int, int]

    @property
    def name(self) -> str:
        return f"L{self.block_id}.{self.module}"


class ImportanceSource:
    """指向已有的重要性数据目录。

    目录结构（沿用 task1 产物）：
        <root>/L{block}_{module}/I_energy.npy
        <root>/L{block}_{module}/I_max_log10.npy
    """

    def __init__(self, root: str | Path, blocks=BLOCKS, modules=MODULES):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"重要性数据目录不存在: {self.root}")
        self.blocks = tuple(blocks)
        self.modules = tuple(modules)
        self._tensors = self._discover()

    def _discover(self) -> dict[str, TensorSpec]:
        tensors = {}
        for block_id in self.blocks:
            for module in self.modules:
                folder = self.root / f"L{block_id}_{module}"
                path = folder / "I_max_log10.npy"
                if not path.exists():
                    raise FileNotFoundError(f"缺少 Smax 文件: {path}")
                arr = np.load(path, mmap_mode="r")
                meta_path = folder / "meta.json"
                shape = (1, 1)
                if meta_path.exists():
                    import json as _json
                    meta = _json.loads(meta_path.read_text(encoding="utf-8"))
                    src_shape = meta.get("shape")
                    if src_shape:
                        shape = (int(src_shape[0]), int(src_shape[1]))
                tensors[f"L{block_id}.{module}"] = TensorSpec(
                    block_id=block_id, module=module,
                    n_weights=int(arr.shape[0]), shape=shape)
                del arr
        return tensors

    @property
    def tensors(self) -> dict[str, TensorSpec]:
        return dict(self._tensors)

    @property
    def total_weights(self) -> int:
        return sum(t.n_weights for t in self._tensors.values())

    def smax_log10(self, name: str):
        spec = self._tensors[name]
        return np.load(self.root / f"L{spec.block_id}_{spec.module}" /
                       "I_max_log10.npy", mmap_mode="r")

    def fingerprint(self) -> dict:
        """记录数据版本指纹（文件大小 + 前 4 MB 的 SHA-256）。"""
        digest = {}
        for name, spec in self._tensors.items():
            path = self.root / f"L{spec.block_id}_{spec.module}" / "I_max_log10.npy"
            h = hashlib.sha256()
            with path.open("rb") as handle:
                h.update(handle.read(1 << 22))
            digest[name] = {"bytes": path.stat().st_size,
                            "sha256_prefix": h.hexdigest()}
        return {"root": str(self.root), "n_tensors": len(self._tensors),
                "total_weights": self.total_weights,
                "coverage_note": COVERAGE_NOTE, "files": digest}

    def describe(self) -> dict:
        return {
            "root": str(self.root),
            "n_tensors": len(self._tensors),
            "total_weights": self.total_weights,
            "tensors": {name: {"n_weights": s.n_weights, "shape": list(s.shape)}
                        for name, s in self._tensors.items()},
            "coverage_note": COVERAGE_NOTE,
        }
