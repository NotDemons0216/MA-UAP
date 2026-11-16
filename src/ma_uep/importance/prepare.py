"""一条龙准备：模型 + 数据集 → F → Smax → 等级图 + 异常遏制阈值。

数据分层（体积由大到小，只有最后两层会被持久化）：

    ① Bit-level 风险地图  16 GB  →  仅 --save-risk-map 时写出
    ② 每权重 Smax        936 MB  →  临时文件，--keep-smax 才保留
    ③ 等级图             234 MB  →  默认产物
    ④ 阈值表              6 KB   →  默认产物
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from ..constants import BLOCKS, COVERAGE_NOTE, MODULES
from ..model.dataset import build_windows, load_text
from ..model.loader import ModelManager
from .activation import collect_F
from .ranker import GlobalRanker
from .scan import scan_all
from .source import ImportanceSource


def prepare(config, repo_root: Path, all_blocks: bool = False,
            save_risk_map: bool = False, keep_smax: bool = False,
            work_dir: str | Path | None = None,
            verbose: bool = True) -> dict:
    started = time.perf_counter()
    blocks = tuple(range(28)) if all_blocks else tuple(config.scope.blocks)
    modules = tuple(config.scope.modules)
    out_root = Path(work_dir) if work_dir else (repo_root / config.output.root / "prepared")
    out_root = Path(out_root)
    smax_dir = out_root / "smax"
    alloc_dir = out_root / "allocation"
    thr_dir = out_root / "thresholds"
    for path in (smax_dir, alloc_dir, thr_dir):
        path.mkdir(parents=True, exist_ok=True)

    # 1) 模型
    manager = ModelManager(config.model.path, config.model.device, repo_root).load()
    specs = list(manager.iter_target_tensors(blocks, modules))
    if verbose:
        print(f"[prepare] 模型已加载；目标 {len(specs)} 个 Tensor "
              f"（{len(blocks)} 个 Block）", flush=True)

    # 2) 数据集
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(manager.path))
    text, source = load_text(config.dataset.split, config.dataset.path)
    eval_set = build_windows(tokenizer, text, config.dataset.max_length,
                             config.dataset.stride)
    eval_set.source = source
    if verbose:
        print(f"[prepare] 数据集 {source.split(':')[0]}，"
              f"{eval_set.n_tokens:,} tokens，{len(eval_set.windows)} 窗口", flush=True)

    # 3) F
    t0 = time.perf_counter()
    F_by_tensor = collect_F(manager, eval_set, specs, verbose=verbose)
    f_seconds = time.perf_counter() - t0
    if verbose:
        print(f"[prepare] F 采集完成（{f_seconds:.0f}s）", flush=True)

    # 4) Smax 扫描
    t0 = time.perf_counter()
    scan_stats = scan_all(manager, specs, F_by_tensor, smax_dir,
                          save_risk_map=save_risk_map)
    scan_seconds = time.perf_counter() - t0
    if verbose:
        print(f"[prepare] Smax 扫描完成（{scan_seconds:.0f}s）", flush=True)

    # 5) 全局排序 + 等级图（复用 GlobalRanker）
    source_obj = ImportanceSource(smax_dir, blocks, modules)
    ranker = GlobalRanker(source_obj)
    thresholds = ranker.thresholds()
    counts = ranker.write_level_maps(thresholds, alloc_dir)

    # 6) 异常遏制阈值
    guard = {name: stats["max_abs"] for name, stats in scan_stats.items()}
    (thr_dir / "tensor_thresholds.json").write_text(json.dumps({
        "model": str(manager.path),
        "dtype": "bfloat16",
        "rule": "nonfinite(w) OR |w| > T_k",
        "action": "set to zero",
        "thresholds": guard,
        "n_tensors": len(guard),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # 7) 元数据
    total_weights = sum(s.n_weights for s in specs)
    high = sum(v["high"] for v in counts.values())
    medium = sum(v["medium"] for v in counts.values())
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": str(manager.path),
        "dataset": {"source": source, "tokens": eval_set.n_tokens,
                    "windows": len(eval_set.windows)},
        "blocks": list(blocks),
        "modules": list(modules),
        "n_tensors": len(specs),
        "total_weights": total_weights,
        "coverage_note": COVERAGE_NOTE,
        "thresholds": thresholds,
        "level_counts": {"high": high, "medium": medium,
                         "low": total_weights - high - medium},
        "timing_seconds": {"collect_F": round(f_seconds, 1),
                           "scan_smax": round(scan_seconds, 1),
                           "total": round(time.perf_counter() - started, 1)},
        "saved_risk_map": save_risk_map,
        "keep_smax": keep_smax,
    }
    (out_root / "prepare_meta.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 8) 按需清理中间产物
    if not keep_smax and not save_risk_map:
        shutil.rmtree(smax_dir, ignore_errors=True)

    if verbose:
        print(f"[prepare] 完成：High {high:,} / Medium {medium:,} / "
              f"Low {total_weights-high-medium:,}", flush=True)
        print(f"[prepare] 产物：{out_root}", flush=True)
    return payload
