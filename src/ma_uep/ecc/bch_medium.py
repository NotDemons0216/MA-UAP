"""Medium 等级：BCH(24,16) 删余码，d=3，纠 1 bit。

删余保留的校验位：Q = (0,2,4,6,8,10,12,14)（冻结）。
校验矩阵 H' = [ I_8 | P_kept ]，其中 P_kept 是母码系统生成矩阵校验部分在保留位置上的行。
单 bit 错误的综合征两两不同且非零，可唯一定位。
"""

from __future__ import annotations

import numpy as np

from ..constants import BCH_MEDIUM
from .bch_high import HighCodec


class MediumCodec:
    level = "medium"
    k = BCH_MEDIUM["k"]
    n = BCH_MEDIUM["n"]
    parity_bits = BCH_MEDIUM["parity_bits"]
    keep_parity = BCH_MEDIUM["keep_parity"]
    distance = BCH_MEDIUM["distance"]
    guaranteed_correction = BCH_MEDIUM["guaranteed_correction"]

    def __init__(self):
        self.mother = HighCodec()
        self.P = self.mother.generator_parity()
        self.P_kept = self.P[list(self.keep_parity), :]
        self.H = np.hstack([np.eye(self.parity_bits, dtype=np.uint8),
                            self.P_kept]).astype(np.uint8)
        self._table = self._build_syndrome_table()
        self._verify_distance()

    # ---------------- 内部 ----------------
    def _build_syndrome_table(self) -> dict:
        table = {}
        for pos in range(self.n):
            syndrome = tuple(int(x) for x in self.H[:, pos])
            if syndrome == (0,) * self.parity_bits:
                raise RuntimeError(f"删余码第 {pos} 列为零列")
            if syndrome in table:
                raise RuntimeError(f"综合征冲突：{pos} 与 {table[syndrome]}")
            table[syndrome] = pos
        return table

    def _verify_distance(self) -> None:
        cols = [tuple(int(x) for x in self.H[:, i]) for i in range(self.n)]
        zero = (0,) * self.parity_bits
        if zero in cols:
            raise RuntimeError("删余码存在零列，d=1")
        if len(set(cols)) != len(cols):
            raise RuntimeError("删余码存在相同列，d=2")
        # 三列异或为 0 ⇒ 存在重量 3 的码字（d=3）
        weight3 = False
        for i in range(self.n):
            for j in range(i + 1, self.n):
                for k in range(j + 1, self.n):
                    if tuple(a ^ b ^ c for a, b, c in
                             zip(cols[i], cols[j], cols[k])) == zero:
                        weight3 = True
                        break
                if weight3:
                    break
            if weight3:
                break
        if not weight3:
            raise RuntimeError("未找到重量 3 的码字，与冻结的 d=3 不符")

    # ---------------- 编码 ----------------
    def encode(self, data_bits):
        data = np.asarray(data_bits, dtype=np.uint8)
        if data.shape != (self.k,):
            raise ValueError(f"数据长度必须为 {self.k}")
        parity = (self.P_kept @ data) % 2
        return [int(x) for x in np.concatenate([parity, data])]

    # ---------------- 译码 ----------------
    def decode(self, received_bits):
        r = np.asarray(received_bits, dtype=np.uint8)
        if r.shape != (self.n,):
            raise ValueError(f"码字长度必须为 {self.n}")
        syndrome = tuple(int(x) for x in (self.H @ r) % 2)
        if syndrome == (0,) * self.parity_bits:
            return self._result(r, "no_error_detected", [], False)
        if syndrome in self._table:
            pos = self._table[syndrome]
            corrected = r.copy()
            corrected[pos] ^= 1
            if ((self.H @ corrected) % 2).any():
                return self._result(r, "correction_check_failed", [pos], True)
            return self._result(corrected, "single_error_corrected", [pos], True)
        return self._result(r, "detected_not_correctable", [], True)

    def _result(self, codeword, status, positions, detected):
        return {
            "decoded_bits": [int(x) for x in codeword[self.parity_bits:]],
            "status": status,
            "corrected_positions": positions,
            "detected_error": detected,
            "codeword_valid": status in ("no_error_detected",
                                         "single_error_corrected"),
        }

    def parameters(self):
        return dict(BCH_MEDIUM)
