"""High 等级：BCH(31,16,7)，完整复用现有母码实现。"""

from __future__ import annotations

import numpy as np

from ..constants import BCH_HIGH
from .vendor import import_bch_code


class HighCodec:
    level = "high"
    k = BCH_HIGH["k"]
    n = BCH_HIGH["n"]
    parity_bits = BCH_HIGH["parity_bits"]
    guaranteed_correction = BCH_HIGH["guaranteed_correction"]

    def __init__(self):
        self._code = None

    @property
    def code(self):
        if self._code is None:
            BCHCode = import_bch_code()
            self._code = BCHCode(BCH_HIGH["m"], BCH_HIGH["primitive_poly"],
                                 BCH_HIGH["t"])
        return self._code

    def encode(self, data_bits):
        return self.code.encode(list(data_bits))

    def decode(self, received_bits):
        result = self.code.decode(list(received_bits))
        return {
            "decoded_bits": list(result.decoded_data),
            "status": result.status,
            "corrected_positions": list(result.error_positions or []),
            "detected_error": result.syndrome_valid is False
            or any(s != 0 for s in result.syndromes),
            "codeword_valid": bool(result.codeword_syndrome_valid),
        }

    def generator_parity(self) -> np.ndarray:
        """系统生成矩阵的校验部分 P（15×16），满足 p = P·u (mod 2)。"""
        P = np.zeros((self.parity_bits, self.k), dtype=np.uint8)
        for j in range(self.k):
            data = [0] * self.k
            data[j] = 1
            P[:, j] = np.array(self.encode(data)[:self.parity_bits], dtype=np.uint8)
        return P

    def parameters(self):
        return dict(BCH_HIGH)
