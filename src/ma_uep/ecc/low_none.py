"""Low 等级：无保护透传（16 bit）。"""

from __future__ import annotations

from ..constants import LOW_NONE


class LowCodec:
    level = "low"
    k = LOW_NONE["k"]
    n = LOW_NONE["n"]
    parity_bits = LOW_NONE["parity_bits"]
    guaranteed_correction = LOW_NONE["guaranteed_correction"]

    def encode(self, data_bits):
        data = list(data_bits)
        if len(data) != self.k:
            raise ValueError(f"数据长度必须为 {self.k}")
        return [int(b) & 1 for b in data]

    def decode(self, received_bits):
        bits = [int(b) & 1 for b in received_bits]
        if len(bits) != self.n:
            raise ValueError(f"码字长度必须为 {self.n}")
        return {
            "decoded_bits": bits,
            "status": "no_protection_passthrough",
            "corrected_positions": [],
            "detected_error": False,
            "codeword_valid": True,
        }

    def parameters(self):
        return dict(LOW_NONE)
