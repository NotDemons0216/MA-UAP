"""信道与故障注入（两段式：整组抽样 + 分块应用）。

设计目标：**与旧实现逐样本同随机流**。

- BER：每个 (Tensor, 等级) 一次 Poisson + 一次 choice，与旧 `inject_errors` 相同；
- Fixed：先在**整张量**范围抽"坏权重"，再按等级依次为每个坏权重抽 bit，
  与旧 `process_tensor_fixed` 相同。

信道只负责抽样，位置→码字的映射与分块应用由 pipeline 完成。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import CHANNEL_EXCLUDED_BF16_BITS


def allowed_positions(n_bits: int, data_offset: int,
                      exclude_bits=CHANNEL_EXCLUDED_BF16_BITS) -> np.ndarray:
    allowed = np.ones(n_bits, dtype=bool)
    for bit in exclude_bits:
        allowed[data_offset + bit] = False
    return allowed


def _flip(words: np.ndarray, word_idx: np.ndarray, bit_idx: np.ndarray) -> None:
    np.bitwise_xor.at(words, word_idx,
                      np.left_shift(np.uint32(1), bit_idx.astype(np.uint32)))


@dataclass
class ChannelConfig:
    mode: str = "ber"
    ber: float = 0.0
    bad_weights: int = 0
    k_bits: int = 1
    seed: int = 42


class BaseChannel:
    mode = "base"

    def __init__(self, config: ChannelConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    # ---------------- 公共 ----------------
    def codec_params(self, n_bits: int, data_offset: int):
        allowed = np.flatnonzero(allowed_positions(n_bits, data_offset))
        return len(allowed), allowed

    def begin_tensor(self, n_weights: int, bad_count: int | None = None) -> None:
        """每个 Tensor 开始时调用一次。bad_count 为本张量的坏权重数。"""

    def plan(self, n_selected: int, n_bits: int, data_offset: int):
        """返回 (组内 word 序号, bit 序号)。"""
        raise NotImplementedError

    def sample(self, n_selected: int, n_bits: int, data_offset: int):
        """便捷封装：返回 (word_idx, bit_idx, n_err)。"""
        word_idx, bit_idx = self.plan(n_selected, n_bits, data_offset)
        return word_idx, bit_idx, int(word_idx.size)

    def apply(self, words: np.ndarray, word_idx: np.ndarray,
              bit_idx: np.ndarray) -> None:
        if word_idx.size:
            _flip(words, word_idx, bit_idx)


class BerChannel(BaseChannel):
    """每个存储位独立以概率 p 翻转。"""

    mode = "ber"

    def plan(self, n_selected: int, n_bits: int, data_offset: int):
        per_word, allowed = self.codec_params(n_bits, data_offset)
        total = n_selected * per_word
        expected = total * self.config.ber
        n_err = int(self.rng.poisson(expected)) if expected > 0 else 0
        if n_err == 0:
            return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
        positions = self.rng.choice(total, size=n_err, replace=False)
        return positions // per_word, allowed[positions % per_word]


class FixedWeightChannel(BaseChannel):
    """固定出错权重数 + 每权重错误位数（整张量范围抽样）。"""

    mode = "fixed"

    def __init__(self, config: ChannelConfig):
        super().__init__(config)
        self.bad_weights_all: np.ndarray | None = None
        self.n_weights = 0

    def begin_tensor(self, n_weights: int, bad_count: int | None = None) -> None:
        self.n_weights = n_weights
        requested = self.config.bad_weights if bad_count is None else bad_count
        n_bad = min(requested, n_weights)
        self.bad_weights_all = (self.rng.choice(n_weights, size=n_bad, replace=False)
                                if n_bad > 0 else np.empty(0, dtype=np.int64))

    def draw_bits(self, count: int, n_bits: int, data_offset: int) -> np.ndarray:
        """为 count 个坏权重依次抽 k 个 bit（顺序与旧实现一致）。"""
        per_word, allowed = self.codec_params(n_bits, data_offset)
        k = min(self.config.k_bits, per_word)
        out = np.empty(count * k, dtype=np.int64)
        cursor = 0
        for _ in range(count):
            pos = self.rng.choice(per_word, size=k, replace=False)
            out[cursor:cursor + k] = allowed[pos]
            cursor += k
        return out

    def plan(self, n_selected: int, n_bits: int, data_offset: int):
        raise RuntimeError("Fixed 模式需由 pipeline 提供等级信息")


def build_channel(config: ChannelConfig):
    if config.mode == "ber":
        return BerChannel(config)
    if config.mode == "fixed":
        return FixedWeightChannel(config)
    raise ValueError(f"未知信道模式 {config.mode!r}")
