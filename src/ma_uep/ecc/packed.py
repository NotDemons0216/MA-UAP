"""向量化打包编解码：把码字放进 uint32，用综合征查表实现批量译码。

与旧实现（ECC/simulation/packed_codec.py）保持逐位等价：
  - 编码查表由同一批标量 codec 生成；
  - 综合征 = H·word mod 2；
  - 纠错查表覆盖重量 ≤ t 的全部图样；
  - 纠正后复核综合征，仍非零则标记为不可纠正。
"""

from __future__ import annotations

import itertools

import numpy as np

from .bch_high import HighCodec
from .bch_medium import MediumCodec

PARITY_LUT = np.array([bin(i).count("1") & 1 for i in range(256)], dtype=np.uint8)

LEVEL_TO_CLASS = {"high": HighCodec, "medium": MediumCodec}


def parity_of(u32: np.ndarray) -> np.ndarray:
    """逐元素偶校验（0/1）。"""
    arr = u32.view(np.uint8).reshape(-1, 4)
    return PARITY_LUT[arr].sum(axis=1) & 1


class PackedCodec:
    def __init__(self, level: str):
        if level not in LEVEL_TO_CLASS:
            raise ValueError(f"PackedCodec 仅支持 high/medium，收到 {level!r}")
        self.level = level
        self.scalar = LEVEL_TO_CLASS[level]()
        self.k = self.scalar.k
        self.n = self.scalar.n
        self.t = self.scalar.guaranteed_correction
        self.encode_table = self._build_encode_table()
        self.H = self._build_check_matrix()
        self.syndrome_table = self._build_syndrome_table()

    # ---------------- 构建 ----------------
    def _build_encode_table(self) -> np.ndarray:
        table = np.zeros(1 << self.k, dtype=np.uint32)
        for value in range(1 << self.k):
            data = [(value >> i) & 1 for i in range(self.k)]
            packed = 0
            for i, bit in enumerate(self.scalar.encode(data)):
                packed |= int(bit) << i
            table[value] = packed
        return table

    def _build_check_matrix(self) -> np.ndarray:
        if self.level == "medium":
            return self.scalar.H.astype(np.uint8)
        G = np.zeros((self.n, self.k), dtype=np.uint8)
        for j in range(self.k):
            packed = int(self.encode_table[1 << j])
            G[:, j] = [(packed >> i) & 1 for i in range(self.n)]
        P = G[:self.scalar.parity_bits, :]
        return np.hstack([np.eye(self.scalar.parity_bits, dtype=np.uint8), P])

    def _build_syndrome_table(self) -> np.ndarray:
        rows = self.H.shape[0]
        table = np.zeros(1 << rows, dtype=np.uint32)
        for weight in range(1, self.t + 1):
            for positions in itertools.combinations(range(self.n), weight):
                err = 0
                syndrome = 0
                for pos in positions:
                    err |= 1 << pos
                    for i in range(rows):
                        syndrome ^= int(self.H[i, pos]) << i
                if table[syndrome] == 0:
                    table[syndrome] = err
        return table

    # ---------------- 编解码 ----------------
    def encode(self, values_u16: np.ndarray) -> np.ndarray:
        return self.encode_table[values_u16]

    def syndromes(self, words: np.ndarray) -> np.ndarray:
        result = np.zeros(len(words), dtype=np.uint16)
        for i in range(self.H.shape[0]):
            mask = 0
            for pos in range(self.n):
                if self.H[i, pos]:
                    mask |= 1 << pos
            bit = parity_of(words & np.uint32(mask))
            result |= bit.astype(np.uint16) << i
        return result

    def decode(self, words: np.ndarray):
        """返回 (corrected_words, status)。status: 0 无错 / 1 已纠正 / 2 不可纠正。"""
        syndromes = self.syndromes(words)
        corrected = words.copy()
        status = np.zeros(len(words), dtype=np.uint8)
        nonzero = syndromes != 0
        if not nonzero.any():
            return corrected, status
        idx = np.flatnonzero(nonzero)
        pattern = self.syndrome_table[syndromes[idx]]
        known = pattern != 0
        corrected[idx[known]] ^= pattern[known]
        status[idx[known]] = 1
        status[idx[~known]] = 2
        if known.any():
            recheck = self.syndromes(corrected[idx[known]])
            bad = recheck != 0
            if bad.any():
                status[idx[known][bad]] = 2
        return corrected, status

    def data_of(self, words: np.ndarray, status=None) -> np.ndarray:
        return ((words >> (self.n - self.k)) & 0xFFFF).astype(np.uint16)
