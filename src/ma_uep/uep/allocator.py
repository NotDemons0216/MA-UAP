"""UEP 分配与开销统计。

分配规则由 GlobalRanker 的阈值完成；本模块负责：
  - 校验"每个权重只属于一个等级"；
  - 统计各级权重数量、校验位、码率、冗余比例。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..constants import BCH_HIGH, BCH_MEDIUM, LEVEL_RATIOS


def load_level_maps(root: str | Path, tensor_names) -> dict:
    root = Path(root)
    maps = {}
    for name in tensor_names:
        key = name.replace(".", "_")          # L0.q_proj -> L0_q_proj
        path = root / key / "level_map.npy"
        maps[name] = np.load(path, mmap_mode="r")
    return maps


def validate(maps: dict) -> dict:
    """检查等级图取值合法、总数一致。"""
    codes = set()
    total = 0
    for name, arr in maps.items():
        n = arr.shape[0]
        total += n
        codes.update(np.unique(np.asarray(arr[::max(1, n // 10000)])).tolist())
    if not codes <= {0, 1, 2}:
        raise ValueError(f"非法等级码: {sorted(codes)}")
    return {"total_weights": total, "level_codes": sorted(codes)}


def allocation_stats(counts: dict, total_weights: int) -> dict:
    """counts: {"high": n, "medium": n, "low": n}"""
    high = int(counts["high"])
    medium = int(counts["medium"])
    low = int(counts["low"])
    if high + medium + low != total_weights:
        raise ValueError("等级数量之和与总权重不符")
    data_bits = 16 * total_weights
    parity_bits = BCH_HIGH["parity_bits"] * high + BCH_MEDIUM["parity_bits"] * medium
    total_bits = data_bits + parity_bits
    return {
        "total_weights": int(total_weights),
        "high": high, "medium": medium, "low": low,
        "high_frac": high / total_weights,
        "medium_frac": medium / total_weights,
        "low_frac": low / total_weights,
        "high_parity_bits": BCH_HIGH["parity_bits"] * high,
        "medium_parity_bits": BCH_MEDIUM["parity_bits"] * medium,
        "data_bits": data_bits,
        "parity_bits": parity_bits,
        "total_bits": total_bits,
        "code_rate": data_bits / total_bits,
        "redundancy": parity_bits / data_bits,
        "frozen_ratios": dict(LEVEL_RATIOS),
    }
