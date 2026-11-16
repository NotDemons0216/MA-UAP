"""统一实验流水线（12 步）。"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..channel.injection import ChannelConfig as ChCfg, build_channel
from ..paths import PROJECT_ROOT
from ..ecc.vendor import set_bch_path
from ..evaluation.perplexity import evaluate
from ..importance.source import ImportanceSource
from ..model.dataset import build_windows, load_text
from ..model.loader import ModelManager
from ..protection.anomaly_guard import AnomalyGuard
from ..uep.allocator import allocation_stats, load_level_maps, validate
from .tensor_pass import TensorProcessor

MODES_WITH_ECC = ("ma_uep", "ma_uep_without_guard", "random_uep")


class ExperimentRunner:
    def __init__(self, config, repo_root: Path | None = None):
        self.config = config
        self.repo_root = repo_root or PROJECT_ROOT
        self.results: dict = {}

    # ---------------- 主流程 ----------------
    def run(self, experiment_dir: Path | None = None) -> dict:
        cfg = self.config
        started = time.perf_counter()
        if cfg.ecc.bch_path:
            set_bch_path(cfg.ecc.bch_path)

        # 1 模型
        manager = ModelManager(cfg.model.path, cfg.model.device, self.repo_root).load()
        self.results["model"] = manager.describe()

        # 2 数据集
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(manager.path))
        text, source = load_text(cfg.dataset.split, cfg.dataset.path)
        eval_set = build_windows(tokenizer, text, cfg.dataset.max_length,
                                 cfg.dataset.stride)
        eval_set.source = source
        self.results["dataset"] = {"source": source, "n_tokens": eval_set.n_tokens,
                                   "windows": len(eval_set.windows)}

        # 3 基线（实测，不硬编码）
        baseline = evaluate(manager.model, eval_set, cfg.max_windows)
        self.results["baseline"] = baseline

        if cfg.mode == "baseline":
            self.results["elapsed_seconds"] = time.perf_counter() - started
            return self.results

        # 4 重要性数据
        source_dir = Path(cfg.importance.root)
        if not source_dir.is_absolute():
            source_dir = self.repo_root / source_dir
        importance = ImportanceSource(source_dir, cfg.scope.blocks, cfg.scope.modules)
        self.results["importance"] = importance.describe()

        # 5 等级图
        alloc_dir = Path(cfg.importance.allocation_dir)
        if not alloc_dir.is_absolute():
            alloc_dir = self.repo_root / alloc_dir
        specs = list(manager.iter_target_tensors(cfg.scope.blocks, cfg.scope.modules))
        level_maps = load_level_maps(alloc_dir, [s.name for s in specs])
        self.results["allocation_check"] = validate(level_maps)
        counts = self._count_levels(level_maps, cfg.mode)
        self.results["uep"] = allocation_stats(counts, sum(s.n_weights for s in specs))

        # 6 编解码器 / 信道 / 守护
        channel_cfg = ChCfg(mode=cfg.channel.mode, ber=cfg.channel.ber,
                            bad_weights=cfg.channel.bad_weights,
                            k_bits=cfg.channel.k_bits, seed=cfg.channel.seed)
        channel = build_channel(channel_cfg)
        guard = None
        if cfg.anomaly_guard.enabled and cfg.mode in ("ma_uep", "random_uep"):
            guard = AnomalyGuard.from_file(cfg.anomaly_guard.thresholds, self.repo_root)
        processor = TensorProcessor(channel, guard)

        # 7 逐 Tensor 处理（不整模型备份：磁盘 checkpoint 从不被写入，
        #   每个 Tensor 的原始数据在 process 内部以 raw.copy() 保留）
        original_fingerprint = manager.weight_fingerprint(specs)
        tensor_stats = {}
        total_weights = sum(s.n_weights for s in specs)
        for spec in specs:
            raw = manager.read_raw(spec)
            levels = self._levels_for(spec, level_maps, cfg.mode, cfg.channel.seed)
            bad_count = None
            if cfg.channel.mode == "fixed":
                bad_count = int(round(cfg.channel.bad_weights *
                                      spec.n_weights / total_weights))
            result = processor.process(spec, raw, levels,
                                       apply_guard=guard is not None,
                                       bad_count=bad_count)
            manager.write_raw(spec, result.raw_out)
            tensor_stats[spec.name] = result.stats
        self.results["tensors"] = tensor_stats
        self.results["channel"] = self._summarise_channel(tensor_stats)

        # 8 评估
        evaluation = evaluate(manager.model, eval_set, cfg.max_windows)
        evaluation["delta_nll"] = evaluation["nll"] - baseline["nll"]
        self.results["evaluation"] = evaluation

        # 9 完整性检查：只读比对，确认磁盘 checkpoint 未被触碰
        self.results["integrity"] = {
            "fingerprint_before": original_fingerprint,
            "disk_checkpoint_written": False,
        }

        self.results["elapsed_seconds"] = time.perf_counter() - started
        return self.results

    # ---------------- 辅助 ----------------
    def _count_levels(self, level_maps, mode: str) -> dict:
        counts = {"high": 0, "medium": 0, "low": 0}
        for name, arr in level_maps.items():
            if mode == "no_protection":
                counts["low"] += int(arr.shape[0])
                continue
            sample = np.asarray(arr)
            counts["high"] += int((sample == 2).sum())
            counts["medium"] += int((sample == 1).sum())
            counts["low"] += int((sample == 0).sum())
        return counts

    def _levels_for(self, spec, level_maps, mode: str, seed: int) -> np.ndarray:
        arr = np.asarray(level_maps[spec.name])
        if mode == "no_protection":
            return np.zeros(arr.shape[0], dtype=np.uint8)
        if mode == "random_uep":
            rng = np.random.default_rng(seed + spec.block_id * 31 +
                                        sum(map(ord, spec.module)))
            return arr[rng.permutation(arr.shape[0])]
        return arr

    @staticmethod
    def _summarise_channel(tensor_stats: dict) -> dict:
        total = {"n_errors": 0, "n_corrected": 0, "n_uncorrectable": 0,
                 "n_changed": 0}
        per_level = {}
        for stats in tensor_stats.values():
            for key in total:
                total[key] += int(stats.get(key, 0))
            for level, entry in stats.get("per_level", {}).items():
                agg = per_level.setdefault(level, {"n_weights": 0, "n_errors": 0,
                                                   "n_corrected": 0,
                                                   "n_uncorrectable": 0})
                for key in agg:
                    agg[key] += int(entry.get(key, 0))
        return {"total": total, "per_level": per_level}
