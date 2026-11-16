"""1024-bit 缩短 BCH 原型（第三阶段 3.2 与 3.3 的共用基础）。

目标配置（来自第三阶段任务书 3.2）
---------------------------------
============  ========================================
域            GF(2^11)
本原多项式     ``p(x) = x^11 + x^2 + 1``
母码长度       ``n0 = 2047``
纠错能力       ``t = 38``
生成多项式     次数应为 **407**
母码信息位     ``k0 = 1640``
缩短方式       把母码**高次侧**的 616 个信息位固定为零（见下方映射说明）
缩短后         ``k = 1024``、``r = 407``、``n = 1431``、码率 ≈ 71.56%
============  ========================================

**缩短映射约定（必须与位序约定一致）**

母码信息多项式 ``m(x)`` 的构造是 ``message = data + [0]*616``：1024 bit 原始信息
在前、616 个零**追加在后**。由于系数列表**低次在前**，这 616 个零位于 ``m`` 的
**高次尾部**，即对应母码"最后" 616 个信息位；它们随后对应母码码字的**高次系数**，
在传输时被省略。于是

* 母码码字 ``c_full``（2047 位）与缩短码字 ``c_short``（1431 位）满足
  ``c_full[i] == c_short[i]``（``i = 0..1430``），
  即 **``c_short`` 就是 ``c_full`` 的前 1431 个系数（低 1431 位）**；
* 系统编码性质在缩短码上仍然成立：``c_short[407:] == 输入数据``（1024 位），
  且 ``c_short[0:407]`` 是校验位；
* ``c_short(α^j) = c_full(α^j)`` 对所有 j 成立（两者是同多項式，``c_full`` 只是次数更高）。

> **命名提醒**：这里的零**不是前缀补零**。曾经把它写成 ``prefix_zeros``/
> "零前缀"是错的：那会让人以为零在低次端（信息序列开头）。
> 正确说法是"在信息序列末尾追加零 = 高次信息尾部补零，并省略对应的高次码字系数"。
> 另曾写错过一次实现（等价于 ``[0]*616 + data``）：那会把数据顶到 ``m`` 的高次端，
> ``c_short[r:]`` 不再是输入数据。现在由
> :meth:`ShortenedBCHCode.assert_shortening_invariant` 在测试中强制校验。

本模块**直接对缩短后的向量**（长度 1431）做 Syndrome、错误定位与翻转，
位置索引即该向量的下标（``0`` 是低次位），因此不需要把缩短码字还原成 2047 位；
同时提供 ``encode_full`` 用于交叉验证上述映射关系。
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from bch import (
    NO_SOLVER_RUN,
    SOLVER_BM,
    SOLVER_NEWTON,
    SOLVER_PGZ,
    BCHCode,
    _xor_gather,
)
from gf2m import poly_degree, poly_to_string, trim

# ``NO_SOLVER_RUN`` 定义在 ``bch`` 中（母码与缩短码共用同一语义），这里重新导出，
# 便于 ``from bch_shortened import NO_SOLVER_RUN`` 的既有代码继续工作。
__all__ = [
    "NO_SOLVER_RUN",
    "ALL_SOLVERS",
    "ShortenedBCHCode",
    "ShortenedDecodeResult",
    "build_shortened_code",
    "build_mother_code",
]

# 全部合法求解器标识（供参数校验与文档使用）。
ALL_SOLVERS = (SOLVER_NEWTON, SOLVER_PGZ, SOLVER_BM)

# 3.2 的目标参数
SHORTENED_M = 11
SHORTENED_PRIMITIVE_POLY = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # x^11 + x^2 + 1
SHORTENED_T = 38
SHORTENED_K = 1024
EXPECTED_GENERATOR_DEGREE = 407
EXPECTED_MOTHER_N = 2047
EXPECTED_MOTHER_K = 1640
# 被省略（不传输）的母码信息位数：616。注意这些零追加在信息序列**末尾**
# （高次尾部），不是低次端的前缀——命名必须反映这一点。
SHORTENED_OMITTED_INFORMATION_BITS = EXPECTED_MOTHER_K - SHORTENED_K  # 616
EXPECTED_SHORTENED_N = SHORTENED_K + EXPECTED_GENERATOR_DEGREE  # 1431


def build_mother_code(verbose: bool = False) -> tuple[BCHCode, float]:
    """构造母码 ``BCH(2047, 1640)``，返回 ``(码对象, 构造耗时秒)``。

    构造过程会由 :class:`bch.BCHCode` 自己完成并自检：分圆陪集、生成多项式次数、
    在全部连续根上取零、整除 ``x^n - 1``、BCH 界。
    """
    started = time.perf_counter()
    code = BCHCode(SHORTENED_M, SHORTENED_PRIMITIVE_POLY, SHORTENED_T)
    elapsed = time.perf_counter() - started
    if verbose:
        print(
            f"  母码构造完成：n0={code.n}, k0={code.k}, r={code.r}, deg g={poly_degree(code.generator_poly)}，"
            f"耗时 {elapsed:.2f} s"
        )
    return code, elapsed


@dataclass
class ShortenedDecodeResult:
    """缩短码的一次解码结果。

    ``status`` / ``codeword_syndrome_valid`` 的语义与 :class:`bch.DecodeResult` 一致：
    ``status`` 只描述解码器做了什么；``codeword_syndrome_valid`` 只说明**输出**满足校验，
    **不说明等于原始发送码字**。真实恢复与否必须由调用方用原始数据比较判定
    （:meth:`ShortenedBCHCode.classify` 提供统一口径）。

    求解器相关字段的语义（刻意区分"请求"与"实际执行"）：

    * ``solver``：**实际执行**的错误定位求解路径，取值 ``"newton_search"`` /
      ``"pgz"``；当 Syndrome 全为 0、没有运行任何求解器时为
      :data:`NO_SOLVER_RUN`（``"not_run"``）。它**不会**写"请求但未执行的求解器"。
    * ``solver_work_units``：该路径的工作量口径（见
      :class:`bch.LocatorSolveResult`）；未运行求解器时为 0。
    * ``solver_seconds``：求解器实际耗时（未运行求解器时为 0.0）。
    """

    received: list[int]
    corrected_codeword: list[int]
    decoded_data: list[int]
    status: str
    syndromes: list[int]
    syndrome_valid: bool
    codeword_syndrome_valid: bool
    error_locator: list[int]
    error_positions: list[int]
    bits_flipped: int
    solver: str
    solver_work_units: int
    failure_reason: str
    solver_seconds: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        import dataclasses

        return dataclasses.asdict(self)


class ShortenedBCHCode:
    """把母码 ``BCH(2^m - 1, k0)`` 缩短到 ``k`` 个信息位的封装。

    参数
    ----
    mother:
        已构造好的母码（应满足 ``mother.n == 2**m - 1``）。
    k:
        缩短后的信息位数（默认 1024）。
    """

    def __init__(self, mother: BCHCode, k: int = SHORTENED_K):
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k 必须是整数。")
        if not 0 < k < mother.k:
            raise ValueError(
                f"缩短后的信息位数 k={k} 必须在 0 与母码 k0={mother.k} 之间（不含两端）。"
            )
        self.mother = mother
        self.k = int(k)
        self.r = mother.r
        self.n = self.k + self.r
        # 被省略（不传输）的母码信息位数 = k0 - k = 616。
        #
        # 命名说明：这些零是**追加在原始信息之后**的（``data + [0]*616``），
        # 由于系数列表是"低次在前"，它们落在母码信息多项式的**高次尾部**，
        # 编码后对应母码码字的**高次系数**（被截断掉、不传输的部分）。
        # 因此不能叫 "prefix"（前缀）——那是把低次端当开头的误解。
        self.omitted_information_bits = mother.k - self.k
        self.decoder_positions = list(range(self.n))

    # ------------------------------------------------------------------
    # 基本属性
    # ------------------------------------------------------------------
    @property
    def code_rate(self) -> float:
        return self.k / self.n

    def parameters(self) -> dict[str, Any]:
        """返回适合写入 CSV 的参数表。"""
        return {
            "m": self.mother.m,
            "primitive_poly": poly_to_string(self.mother.field.primitive_poly),
            "mother_n": self.mother.n,
            "mother_k": self.mother.k,
            "mother_r": self.mother.r,
            "t": self.mother.t,
            "design_distance": self.mother.design_distance,
            "generator_degree": poly_degree(self.mother.generator_poly),
            "omitted_information_bits": self.omitted_information_bits,
            "shortened_k": self.k,
            "shortened_r": self.r,
            "shortened_n": self.n,
            "shortened_code_rate": self.code_rate,
            "shortened_redundancy_ratio": 1 / self.code_rate - 1,
            "syndrome_count": 2 * self.mother.t,
        }

    # ------------------------------------------------------------------
    # 输入校验
    # ------------------------------------------------------------------
    def _validate_bits(self, bits: Sequence[int], expected: int, name: str) -> list[int]:
        if isinstance(bits, (str, bytes, bytearray)) or not isinstance(bits, Sequence):
            raise TypeError(f"{name} 必须是 0/1 组成的序列。")
        if len(bits) != expected:
            raise ValueError(
                f"{name} 长度错误：期望 {expected} bit，实际 {len(bits)} bit。"
            )
        out: list[int] = []
        for index, value in enumerate(bits):
            if isinstance(value, bool):
                out.append(int(value))
            elif isinstance(value, int) and value in (0, 1):
                out.append(value)
            else:
                raise ValueError(f"{name}[{index}]={value!r} 不是二进制值 0 或 1。")
        return out

    def _validate_data(self, data_bits: Sequence[int]) -> list[int]:
        return self._validate_bits(data_bits, self.k, "data_bits")

    def _validate_transmitted(self, received: Sequence[int]) -> list[int]:
        return self._validate_bits(received, self.n, "received")

    @staticmethod
    def _validate_solver(solver: str) -> str:
        """校验 ``solver`` 参数；非法取值**明确报错，不静默回退**。"""
        if solver not in ALL_SOLVERS:
            raise ValueError(
                f"不支持的求解器 {solver!r}；可用取值："
                f"{SOLVER_NEWTON!r}（Newton 方程搜索）、{SOLVER_PGZ!r}（PGZ 参考实现）、"
                f"{SOLVER_BM!r}（经典 Berlekamp–Massey 迭代）。"
            )
        return solver

    def _validate_positions(self, positions: Sequence[int] | None) -> list[int]:
        """严格校验允许搜索/翻转的位置集合（与母码同一套规则）。"""
        if positions is None:
            return list(range(self.n))
        if isinstance(positions, (str, bytes, bytearray)) or not isinstance(
            positions, Sequence
        ):
            raise TypeError("positions 必须是整数序列（例如 range 或 list）。")
        checked: list[int] = []
        for index, raw in enumerate(positions):
            if isinstance(raw, bool):
                raise TypeError(
                    f"positions[{index}]={raw!r} 是布尔值；位置必须是整数下标。"
                )
            if not isinstance(raw, int):
                raise TypeError(
                    f"positions[{index}]={raw!r} 不是整数；位置必须是整数下标。"
                )
            if not 0 <= raw < self.n:
                raise ValueError(
                    f"positions[{index}]={raw} 超出合法范围 0..{self.n - 1}。"
                )
            checked.append(raw)
        return sorted(set(checked))

    # ------------------------------------------------------------------
    # 编码
    # ------------------------------------------------------------------
    def encode_full(self, data_bits: Sequence[int]) -> list[int]:
        """按"母码长度 k0" 编码：``message = data + [0]*616`` → 2047 bit 母码字。

        零追加在信息序列**末尾**：由于系数列表低次在前，它们对应母码信息多项式的
        **高次尾部系数**（即母码的"最后" 616 个信息位）。这样截取母码码字的
        **低 1431 位**（前 1431 个系数）后，缩短码字仍是系统码：``c_short[r:] == data``。
        本方法主要用于**交叉验证缩短映射**，日常使用请用 :meth:`encode`。
        """
        data = self._validate_data(data_bits)
        message = data + [0] * self.omitted_information_bits
        return self.mother.encode(message)

    def encode(self, data_bits: Sequence[int]) -> list[int]:
        """缩短编码：``k`` 个信息位 → ``n = k + r`` 个码字位（低次优先）。

        实现上先按母码编码，再**截取母码码字的前 n 个系数（低 n 位）**，
        等价于"把追加在信息末尾的 ``omitted_information_bits`` 个高位信息位固定为零、
        并在传输时省略它们对应的高次码字系数"。该等价性由
        :meth:`assert_shortening_invariant` 交叉验证。
        """
        return self.encode_full(data_bits)[: self.n]

    def extract_data(self, codeword: Sequence[int]) -> list[int]:
        """从缩短码字中取出信息位（下标 ``r..n-1``）。"""
        bits = self._validate_transmitted(codeword)
        return bits[self.r :]

    def assert_shortening_invariant(self, data_bits: Sequence[int]) -> None:
        """验证缩短映射的两条不变式（供测试与自检调用）。"""
        full = self.encode_full(data_bits)
        short = self.encode(data_bits)
        # 1. 缩短码字 = 母码码字的低 n 个系数
        if full[: self.n] != short:
            raise AssertionError("缩短映射不成立：母码低 n 位与缩短码字不一致。")
        # 2. 母码高位为 0 的位置正是被缩短掉的信息位（不被传输）
        # 3. 数据位位于高次：short[r:] == 输入数据
        if short[self.r :] != list(data_bits):
            raise AssertionError("系统编码不成立：缩短码字的高次位不是输入数据。")

    # ------------------------------------------------------------------
    # Syndrome
    # ------------------------------------------------------------------
    def syndromes(self, received: Sequence[int]) -> list[int]:
        """计算 ``S_j = r(α^j)``，``j = 1..2t``（共 76 个）。

        **优化后**：复用母码的预计算幂表 ``α^{j·p}``，并且只对**非零位置**做异或收集
        （``S_j = Σ_p r_p·α^{j·p}``，二进制下"乘 bit"就是"是否计入异或"）。
        数学定义与原来完全一致（原 Horner 参考实现保留在
        :meth:`bch.BCHCode._eval_at_power`，测试中逐项对照）。

        注意：``r(x)`` 的次数是 ``n-1``，与"补零后的母码多项式"只差非零因子
        ``α^{-616j}``，对是否为零等价；这一点与优化无关，未改变。
        """
        bits = self._validate_transmitted(received)
        # 母码的幂表按位置索引，缩短码的位置范围 0..n-1 是它的子集，可直接复用。
        rows = self.mother._syndrome_power_rows()
        set_positions = [position for position, bit in enumerate(bits) if bit]
        return [_xor_gather(row, set_positions) for row in rows]

    # ------------------------------------------------------------------
    # 解码
    # ------------------------------------------------------------------
    def decode(
        self,
        received: Sequence[int],
        positions: Sequence[int] | None = None,
        solver: str = SOLVER_NEWTON,
    ) -> ShortenedDecodeResult:
        """缩短 BCH 解码：Syndrome → 求 σ → Chien → 翻转 → 复核。

        ``positions`` 默认是**全部实际传输位置** ``0..n-1``；
        **绝不包含被缩短掉、固定为零的母码信息位**（这正是缩短码解码的关键约束）。

        ``solver`` **真正决定执行哪条求解路径**：``SOLVER_NEWTON``（默认）走
        ``solve_error_locator_newton_search``，``SOLVER_PGZ`` 走 ``solve_error_locator_pgz``，
        ``SOLVER_BM`` 走经典 BM ``solve_error_locator_bm``；
        非法取值会直接抛 ``ValueError``，**不会静默回退**。
        结果里的 ``solver`` 字段只反映**实际执行**的路径；
        当 Syndrome 全为 0、没有运行求解器时，该字段为 ``NO_SOLVER_RUN``。
        """
        self._validate_solver(solver)
        bits = self._validate_transmitted(received)
        allowed = self._validate_positions(positions)
        syndromes = self.syndromes(bits)

        if not any(syndromes):
            # 没有运行任何求解器：元数据必须如实反映这一点，而不是写"请求的求解器"。
            return self._result(
                received=bits,
                corrected=bits[:],
                status="no_error_detected",
                syndromes=syndromes,
                error_locator=[1],
                error_positions=[],
                codeword_syndrome_valid=True,
                solver=NO_SOLVER_RUN,
                solver_work_units=0,
                failure_reason="",
                solver_seconds=0.0,
                notes=[
                    "Syndrome 全为 0，未做任何翻转，也未运行错误定位求解器"
                    f"（请求的求解器为 {solver!r}）。"
                ],
            )

        started = time.perf_counter()
        if solver == SOLVER_NEWTON:
            solved = self.mother.solve_error_locator_newton_search(
                syndromes, positions=allowed
            )
        elif solver == SOLVER_BM:
            solved = self.mother.solve_error_locator_bm(
                syndromes, positions=allowed
            )
        else:  # SOLVER_PGZ（合法性已由 _validate_solver 保证）
            solved = self.mother.solve_error_locator_pgz(syndromes)
        solver_seconds = time.perf_counter() - started
        locator = solved.error_locator
        degree = poly_degree(locator)
        # **Chien 根搜索复用**：newton_search / berlekamp_massey 在判断"次数 = 根数"时
        # 已在**同一个** allowed 集合上完整搜过一次根，直接复用其结果；
        # PGZ 内部不做根搜索（error_positions=None），这里照旧自行搜索。
        reused_chien = solved.error_positions is not None
        if reused_chien:
            error_positions = list(solved.error_positions)  # type: ignore[arg-type]
        else:
            error_positions, _ = self.mother.chien_search(locator, allowed)

        failure_reason = ""
        if not solved.solved:
            failure_reason = f"错误定位多项式求解未收敛：{solved.detail}"
        elif degree > self.mother.t:
            failure_reason = (
                f"错误定位多项式次数 {degree} 超过纠错能力 t={self.mother.t}，"
                "错误个数超出设计能力。"
            )
        elif degree != len(error_positions):
            failure_reason = (
                f"错误定位多项式次数 {degree} 与在允许范围内找到的根数 "
                f"{len(error_positions)} 不一致，无法得到一致的错误位置集合。"
            )
        elif not error_positions:
            failure_reason = "错误定位多项式在允许的位置范围内没有根。"

        if failure_reason:
            return self._result(
                received=bits,
                corrected=bits[:],
                status="decode_failure",
                syndromes=syndromes,
                error_locator=locator,
                error_positions=[],
                codeword_syndrome_valid=False,
                solver=solved.solver,
                solver_work_units=solved.work_units,
                failure_reason=failure_reason,
                solver_seconds=solver_seconds,
            )

        corrected = bits[:]
        for position in error_positions:
            corrected[position] ^= 1
        recomputed = self.syndromes(corrected)
        if any(recomputed):
            return self._result(
                received=bits,
                corrected=corrected,
                status="correction_check_failed",
                syndromes=syndromes,
                error_locator=locator,
                error_positions=error_positions,
                codeword_syndrome_valid=False,
                solver=solved.solver,
                solver_work_units=solved.work_units,
                failure_reason="翻转后重新计算的 Syndrome 仍不为零。",
                solver_seconds=solver_seconds,
                notes=[f"复核失败后的 Syndrome：{recomputed}"],
            )

        return self._result(
            received=bits,
            corrected=corrected,
            status="corrected",
            syndromes=syndromes,
            error_locator=locator,
            error_positions=error_positions,
            codeword_syndrome_valid=True,
            solver=solved.solver,
            solver_work_units=solved.work_units,
            failure_reason="",
            solver_seconds=solver_seconds,
        )

    def _result(self, **kwargs: Any) -> ShortenedDecodeResult:
        """组装结果，并补齐可由其它字段推出的量。

        * ``syndrome_valid``：**输入**接收字的 Syndrome 是否全为 0（未显式给出时自动推导）；
        * ``solver_work_units`` 由调用方给出（口径随求解器不同）。
        """
        received = kwargs.pop("received")
        corrected = kwargs.pop("corrected")
        kwargs.setdefault("syndrome_valid", not any(kwargs.get("syndromes", [])))
        kwargs.setdefault("bits_flipped", len(kwargs.get("error_positions", [])))
        return ShortenedDecodeResult(
            received=received,
            corrected_codeword=corrected,
            decoded_data=self.extract_data(corrected),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # 分类（必须传入原始数据）
    # ------------------------------------------------------------------
    def classify(self, original_data: Sequence[int], result: ShortenedDecodeResult) -> str:
        """把解码结果与**原始数据**比较后分类（与母码同一套口径）。"""
        original = self._validate_data(original_data)
        recovered = result.decoded_data == original
        if recovered:
            if result.status == "no_error_detected":
                return "no_error_no_change"
            return "correctly_recovered"
        if result.status == "decode_failure":
            return "decode_failure_not_recovered"
        if result.status == "correction_check_failed":
            return "correction_check_failed_wrong_data"
        if result.bits_flipped > 0:
            return "miscorrected_wrong_data"
        return "valid_codeword_wrong_data"

    def describe(self) -> str:
        return (
            f"缩短 BCH({self.n},{self.k})，母码 BCH({self.mother.n},{self.mother.k}) "
            f"于 GF(2^{self.mother.m})，t={self.mother.t}，r={self.r}，"
            f"省略母码高次信息位 {self.omitted_information_bits} bit，"
            f"码率 {self.code_rate:.4%}"
        )


def _horner_over_alpha_power(field, bits: Sequence[int], power: int) -> int:
    """Horner 法求 ``r(α^power)``（``bits`` 为低次优先系数）。"""
    factor = field.value_from_power(power)
    result = 0
    for coefficient in reversed(bits):
        result = field.mul(result, factor)
        if coefficient:
            result ^= 1
    return result


def build_shortened_code(verbose: bool = False) -> tuple[ShortenedBCHCode, float]:
    """构造 3.2 的目标缩短码（母码构造 + 参数校验）。"""
    mother, elapsed = build_mother_code(verbose=verbose)
    degree = poly_degree(mother.generator_poly)
    if degree != EXPECTED_GENERATOR_DEGREE:
        raise AssertionError(
            f"生成多项式次数为 {degree}，与任务书预期 {EXPECTED_GENERATOR_DEGREE} 不符。"
            "请先检查有限域实现、根集合与分圆陪集，不要绕过该验证。"
        )
    if mother.n != EXPECTED_MOTHER_N or mother.k != EXPECTED_MOTHER_K:
        raise AssertionError(
            f"母码参数 (n0,k0)=({mother.n},{mother.k}) 与预期 "
            f"({EXPECTED_MOTHER_N},{EXPECTED_MOTHER_K}) 不符。"
        )
    code = ShortenedBCHCode(mother, SHORTENED_K)
    if code.n != EXPECTED_SHORTENED_N:
        raise AssertionError(
            f"缩短后码长 {code.n} 与预期 {EXPECTED_SHORTENED_N} 不符。"
        )
    return code, elapsed


if __name__ == "__main__":  # pragma: no cover - 手工检查用
    code, build_seconds = build_shortened_code(verbose=True)
    print(code.describe())
    for key, value in code.parameters().items():
        if key in ("generator_poly_low_first",):
            continue
        print(f"  {key} = {value}")
