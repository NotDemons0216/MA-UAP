"""逐权重 Smax 扫描（分块、流式，不需要保存 Bit-level 风险地图）。

对每个权重 w：
    S(w, b) = F_k · |ΔW(w, b)|，  ΔW = bf16(raw ^ 2^b) − bf16(raw)
    Smax(w) = max_{b ∈ B} S(w, b)，B = {0..15} \\ {14}

输出（与 task1 产物布局兼容，供 GlobalRanker 直接使用）：
    <out>/L{block}_{module}/I_max_log10.npy   float32, log10(Smax)
同时统计每个 Tensor 的 max|w|（异常遏制阈值）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..constants import BF16_BITS, EXCLUDED_BF16_BITS

RISK_BITS = tuple(b for b in range(BF16_BITS) if b not in EXCLUDED_BF16_BITS)
CHUNK = 200_000


def _bf16_to_float(raw: np.ndarray) -> np.ndarray:
    """int16 原始位 → float64，不经过 FP32 数值拆分。"""
    return torch.from_numpy(raw.astype(np.int16)).view(torch.bfloat16).double().numpy()


def scan_tensor(model_manager, spec, F: np.ndarray, out_dir: Path,
                save_risk_map: bool = False, chunk: int = CHUNK,
                verbose: bool = False) -> dict:
    """返回 {name, n_weights, max_abs, seconds}。"""
    raw = model_manager.read_raw(spec)
    n = len(raw)
    K = int(F.shape[0])
    log_map = None
    if save_risk_map:
        log_map = np.lib.format.open_memmap(
            out_dir / "risk_log10.npy", mode="w+", dtype=np.float32,
            shape=(n, len(RISK_BITS)))

    smax = np.empty(n, dtype=np.float64)
    max_abs = 0.0
    import time
    started = time.perf_counter()
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block_raw = raw[start:end]
        base = _bf16_to_float(block_raw)
        col = np.arange(start, end) % K
        fk = F[col]
        best = None
        for bit_index, bit in enumerate(RISK_BITS):
            flipped = (block_raw.view(np.uint16) ^ np.uint16(1 << bit)).view(np.int16)
            delta = np.abs(_bf16_to_float(flipped) - base)
            s = fk * delta
            best = s if best is None else np.maximum(best, s)
            if log_map is not None:
                with np.errstate(divide="ignore"):
                    log_map[start:end, bit_index] = np.log10(
                        s, where=s > 0, out=np.full(s.shape, -np.inf))
            del flipped, delta, s
        smax[start:end] = best
        max_abs = max(max_abs, float(np.abs(base).max()))
        del base, best, fk, col
        if verbose:
            print(f"    {spec.name} {end}/{n}", flush=True)

    target = out_dir
    target.mkdir(parents=True, exist_ok=True)
    with np.errstate(divide="ignore"):
        np.save(target / "I_max_log10.npy",
                np.log10(smax, where=smax > 0,
                         out=np.full(smax.shape, -np.inf)).astype(np.float32))
    if log_map is not None:
        log_map.flush()
        del log_map
    meta = {"name": spec.name, "block_id": spec.block_id,
            "module": spec.module, "n_weights": int(n), "K": K,
            "max_abs": max_abs, "shape": list(spec.shape),
            "seconds": round(time.perf_counter() - started, 1)}
    return meta


def scan_all(model_manager, specs, F_by_tensor: dict, out_dir: str | Path,
             save_risk_map: bool = False) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for spec in specs:
        target = out_dir / f"L{spec.block_id}_{spec.module}"
        target.mkdir(parents=True, exist_ok=True)
        meta = scan_tensor(model_manager, spec, F_by_tensor[spec.name], target,
                           save_risk_map=save_risk_map)
        stats[spec.name] = meta
        print(f"  {spec.name}: {meta['n_weights']:,} 权重，"
              f"{meta['seconds']}s，max|w|={meta['max_abs']:.6f}", flush=True)
    return stats
