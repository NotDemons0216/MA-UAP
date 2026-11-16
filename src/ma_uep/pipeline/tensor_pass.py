"""唯一的张量遍历实现（整组抽样 + 分块应用）。

- BER：每 (Tensor, 等级) 一次抽样，与旧实现同随机流；
- Fixed：整张量抽"坏权重"，按等级依次抽 bit，与旧实现同随机流；
- 应用阶段按 chunk 进行，峰值内存受限于单个 chunk。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..constants import LEVEL_CODES
from ..ecc.packed import PackedCodec

LEVEL_ITERATION_ORDER = ("low", "medium", "high")
CODEWORD_BITS = {"low": 16, "medium": 24, "high": 31}
CHUNK = 1_000_000


@dataclass
class TensorResult:
    name: str
    raw_out: np.ndarray
    stats: dict = field(default_factory=dict)


class TensorProcessor:
    def __init__(self, channel, guard=None, chunk: int = CHUNK):
        self.channel = channel
        self.guard = guard
        self.chunk = chunk
        self._codecs: dict[str, PackedCodec] = {}

    def codec(self, level: str) -> PackedCodec:
        if level not in self._codecs:
            self._codecs[level] = PackedCodec(level)
        return self._codecs[level]

    # ---------------- 主入口 ----------------
    def process(self, spec, raw_u16: np.ndarray, levels: np.ndarray,
                apply_guard: bool = True, bad_count: int | None = None) -> TensorResult:
        out = raw_u16.copy()
        stats = {"tensor": spec.name, "n_weights": int(len(raw_u16)),
                 "n_errors": 0, "n_corrected": 0, "n_uncorrectable": 0,
                 "n_detected": 0, "per_level": {}}
        self.channel.begin_tensor(len(levels), bad_count)

        boundaries = list(range(0, len(levels), self.chunk)) + [len(levels)]
        for level in LEVEL_ITERATION_ORDER:
            code = LEVEL_CODES[level]
            n_bits = CODEWORD_BITS[level]
            entry = {"n_weights": 0, "codeword_bits": n_bits,
                     "n_errors": 0, "n_corrected": 0,
                     "n_uncorrectable": 0, "n_detected": 0}

            # 1) 每个 chunk 的目标数量
            counts = [int((levels[s:e] == code).sum())
                      for s, e in zip(boundaries[:-1], boundaries[1:])]
            n_selected = sum(counts)
            if n_selected == 0:
                continue
            cumulative = np.cumsum([0] + counts)

            # 2) 抽样
            if self.channel.mode == "fixed":
                word_idx, bit_idx = self._plan_fixed(levels, code, n_bits)
            else:
                word_idx, bit_idx = self.channel.plan(n_selected, n_bits,
                                                      n_bits - 16)

            # 3) 分块应用
            if word_idx.size:
                chunk_of = np.searchsorted(cumulative, word_idx, side="right") - 1
            else:
                chunk_of = np.empty(0, dtype=np.int64)
            for j, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                sel = np.flatnonzero(levels[start:end] == code)
                if sel.size == 0:
                    continue
                mask = chunk_of == j
                entry["n_weights"] += int(sel.size)
                self._process_chunk(level, raw_u16, sel + start, out, entry,
                                    word_idx[mask] - cumulative[j], bit_idx[mask])
                del sel, mask

            entry["n_errors"] = int(word_idx.size)
            stats["per_level"][level] = entry
            for key in ("n_errors", "n_corrected", "n_uncorrectable", "n_detected"):
                stats[key] += entry[key]

        stats["n_changed"] = int((out != raw_u16).sum())
        if self.guard is not None and apply_guard:
            out, guard_stats = self.guard.apply_by_level(out, spec.name, levels)
            stats["anomaly_guard"] = guard_stats
            stats["n_changed"] = int((out != raw_u16).sum())
        return TensorResult(name=spec.name, raw_out=out, stats=stats)

    # ---------------- Fixed 模式 ----------------
    def _plan_fixed(self, levels, code, n_bits):
        bad = self.channel.bad_weights_all
        if bad is None or bad.size == 0:
            return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
        in_level = levels[bad] == code
        selected = bad[in_level]
        if selected.size == 0:
            return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
        ranks = self._group_ranks(levels, code, selected)
        bits = self.channel.draw_bits(selected.size, n_bits, n_bits - 16)
        k = bits.size // selected.size
        word_idx = np.repeat(ranks, k)
        return word_idx, bits

    def _group_ranks(self, levels, code, global_indices) -> np.ndarray:
        """把全局下标映射为其在"该等级组内"的序号（顺序保持不变）。"""
        ranks = np.empty(len(global_indices), dtype=np.int64)
        order = np.argsort(global_indices)
        sorted_idx = global_indices[order]
        running = 0
        n = len(levels)
        cursor = 0
        for start in range(0, n, self.chunk):
            end = min(start + self.chunk, n)
            pos = np.searchsorted(sorted_idx, [start, end])
            lo, hi = int(pos[0]), int(pos[1])
            if lo == hi:
                running += int((levels[start:end] == code).sum())
                continue
            members = np.flatnonzero(levels[start:end] == code) + start
            local = sorted_idx[lo:hi]
            ranks[order[lo:hi]] = running + np.searchsorted(members, local)
            running += members.size
            del members
        return ranks

    # ---------------- 单 chunk ----------------
    def _process_chunk(self, level, raw_u16, idx, out, entry,
                       word_idx, bit_idx) -> None:
        if level == "low":
            words = raw_u16[idx].astype(np.uint32)
            self.channel.apply(words, word_idx, bit_idx)
            out[idx] = words.astype(np.uint16)
            return

        codec = self.codec(level)
        words = codec.encode(raw_u16[idx])
        self.channel.apply(words, word_idx, bit_idx)
        corrected, status = codec.decode(words)
        out[idx] = codec.data_of(corrected, status)
        entry["n_corrected"] += int((status == 1).sum())
        entry["n_uncorrectable"] += int((status == 2).sum())
        entry["n_detected"] += int((status == 2).sum())
