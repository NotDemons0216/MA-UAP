"""分等级统一调度接口。"""

from __future__ import annotations

from ..constants import LEVEL_ORDER
from .bch_high import HighCodec
from .bch_medium import MediumCodec
from .low_none import LowCodec


class LevelCodec:
    """按等级编码 / 译码单个 16 bit 权重。"""

    def __init__(self):
        self.codecs = {"high": HighCodec(), "medium": MediumCodec(),
                       "low": LowCodec()}

    def encode_weight(self, data_bits, level):
        codec = self._get(level)
        return codec.encode(data_bits)

    def decode_weight(self, received_bits, level):
        codec = self._get(level)
        result = codec.decode(received_bits)
        result["level"] = level
        return result

    def _get(self, level):
        if level not in self.codecs:
            raise ValueError(f"未知等级 {level!r}，可选 {LEVEL_ORDER}")
        return self.codecs[level]

    def codeword_length(self, level) -> int:
        return self._get(level).n

    def describe(self) -> dict:
        return {name: {"n": c.n, "k": c.k, "parity": c.parity_bits,
                       "guaranteed_correction": c.guaranteed_correction}
                for name, c in self.codecs.items()}
