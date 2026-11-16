"""二进制窄义原始 BCH 码：生成多项式构造、系统编码、Syndrome、错误定位、Chien 搜索。

第三阶段 3.1 的实现范围
----------------------
本模块实现**原始（非缩短）窄义二进制 BCH 码**的核心算法：

* ``BCH(31, 21)``，纠错能力 ``t = 2``，生成多项式次数 10，域 GF(2^5)、``p(x) = x^5 + x^2 + 1``；
* ``BCH(31, 16)``，纠错能力 ``t = 3``，生成多项式次数 15。

生成多项式、根集合、陪集归并都由程序构造并自检，代码中不硬编码已知结果。

数学约定（编码器/解码器全程一致）
--------------------------------
1. **系数列表低次优先**：``poly[i]`` 是 ``x^i`` 的系数。
2. **位序**：``codeword[i]`` 对应 ``x^i``，即 ``codeword[0]`` 是最低次位。
3. **系统编码**：``c(x) = x^r·m(x) + (x^r·m(x) mod g(x))``，``r = deg g``。
   于是 ``codeword[0..r-1]`` 是校验位，``codeword[r..n-1]`` 是数据位。
4. **Syndrome**：``S_j = r(α^j)``，``j = 1..2t``（返回列表下标 0 对应 ``S_1``）。
   ``S_0`` 不在编码约束内。
5. **错误定位多项式**：``σ(x) = ∏_l (1 + X_l·x)``，``X_l = α^{p_l}``（``p_l`` 是错误所在的
   系数下标），``σ_0 = 1``，``deg σ = 错误个数``。它满足 **Newton 恒等式**

       ``σ_0·S_k + σ_1·S_{k-1} + ... + σ_v·S_{k-v} = 0``   （``k > v``）

   这是本模块求 σ 的唯一数学依据，测试中逐位验证。
6. **Chien Search**：错误位于系数 ``x^p`` ⟺ ``σ(α^{-p}) = 0``。
7. **状态与结果分离**：``status`` 只描述解码器做了什么；``codeword_syndrome_valid``
   只表示输出码字满足校验关系。**没有任何字段声称"恢复了原始数据"**——
   那必须由调用方（测试/实验脚本）拿原始数据比较，``classify()`` 提供统一口径。

两个错误定位求解器（命名必须与实现一致）
----------------------------------------
``solve_error_locator_newton_search`` 与 ``solve_error_locator_pgz`` 都只依赖上面第 5 条的
Newton 恒等式，**共用同一个高斯消元模块**，区别只在"搜索顺序与接受条件"：

* ``newton_search``：按错误个数递增筛选，只解当前规模，命中即返回；
* ``pgz``：按错误个数递减筛选，逐个规模求解。

因此二者的"一致性检查"是**实现内部互检**，能增强可信度，
但**不等同于两个真正独立算法之间的交叉验证**。
本项目当前**没有**实现经典 Berlekamp–Massey 迭代；将来若加入，
应新增独立的求解器标识（例如 ``berlekamp_massey``），并单独做跨算法验证。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence
from gf2m import GF2m, poly_degree, poly_mod, poly_to_string, trim
from lin_alg import (
    SOLUTION_INCONSISTENT,
    SOLUTION_UNDERDETERMINED,
    SOLUTION_UNIQUE,
    solve_linear_system,
)

# ---------------------------------------------------------------- 状态与分类
# 解码动作状态
STATUS_NO_ERROR = "no_error_detected"
STATUS_CORRECTED = "corrected"
STATUS_DECODE_FAILURE = "decode_failure"
# 翻转后重新计算 Syndrome 仍不通过：不得报告普通的 corrected。
STATUS_CORRECTION_CHECK_FAILED = "correction_check_failed"

# 与原始数据比较后的分类（由 `classify` 给出）
CLASS_CORRECTED = "correctly_recovered"
CLASS_NO_ERROR = "no_error_no_change"
CLASS_MISCORRECTED = "miscorrected_wrong_data"
CLASS_VERIFY_FAILED = "correction_check_failed_wrong_data"
CLASS_DECODE_FAILURE = "decode_failure_not_recovered"
CLASS_VALID_WRONG = "valid_codeword_wrong_data"

# 求解器名称。
#
# 三条错误定位路径，命名与实现严格对应：
#   * ``berlekamp_massey``：**经典 BM 迭代**（LFSR 综合，带 b 归一化）。
#     这是与另外两条**真正不同的算法**：它不建立、也不求解任何线性方程组。
#   * ``newton_search``：按候选错误个数**递增**筛选 Newton 恒等式线性方程组 + 三重自洽校验。
#   * ``pgz``：Peterson–Gorenstein–Zierler，按候选错误个数**递减**逐个解方程。
#
# ``newton_search`` 与 ``pgz`` **共用同一类 Newton 方程与同一个高斯消元模块**，
# 它们之间的一致性是"实现内部互检"；而 BM 与它们的一致性才是**跨算法**的独立验证。
SOLVER_NEWTON = "newton_search"
SOLVER_PGZ = "pgz"
SOLVER_BM = "berlekamp_massey"

# 解码结果的 ``solver`` 元数据：当 Syndrome 全为 0、**没有运行任何求解器**时使用。
# 目的：不让 ``solver`` 字段把"请求使用的求解器"冒充成"实际执行的求解器"。
NO_SOLVER_RUN = "not_run"


@dataclass
class LocatorSolveResult:
    """错误定位多项式的一次求解结果。

    ``work_units`` 的口径**随求解器而不同**，三者**不可当作统一的运算次数比较**：

    * ``solver == "berlekamp_massey"``：BM 主循环中**发生多项式更新**（Δ≠0）的次数，
      上限为 ``2t``，通常约为 ``2v``；
    * ``solver == "newton_search"``：尝试过的候选规模个数（≤ ``t``）；
    * ``solver == "pgz"``：尝试过的线性方程组规模之和（``Σ(2t - v)``）。

    它们也都不代表硬件延迟或门级运算量。

    ``error_positions`` 为**可选的复用字段**：
    ``newton_search`` 与 ``berlekamp_massey`` 在判断"次数 = 根数"时已经在该位置集合上
    完整做过一次 Chien 搜索，因此把它们找到的位置一并返回，供解码流程**直接复用**，
    避免对同一个 σ 再做一次相同的根搜索。
    ``pgz`` 内部不做根搜索，故为 ``None``（解码器仍会自己调用 Chien）。
    """

    solver: str
    error_locator: list[int]
    error_count: int
    solved: bool
    work_units: int
    equations_used: int
    detail: str = ""
    error_positions: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DecodeResult:
    """一次 BCH 解码的结构化结果。

    字段语义（刻意区分"操作"与"结果"）：

    * ``status``：解码器实际做了什么，取值
      ``no_error_detected`` / ``corrected`` / ``correction_check_failed`` / ``decode_failure``；
    * ``syndrome_valid``：**输入**接收字的 Syndrome 是否全为 0；
    * ``codeword_syndrome_valid``：**输出**码字的 Syndrome 是否全为 0；
      它**只说明满足校验关系，不说明等于原始发送码字**；
    * ``error_positions`` / ``bits_flipped``：解码器实际翻转了哪些位、翻了几位；
    * ``solver`` / ``solver_work_units``：使用的求解器与其工作量口径（见
      :class:`LocatorSolveResult`）。
    """

    received: list[int]
    corrected_codeword: list[int]
    decoded_data: list[int]
    status: str
    syndromes: list[int]
    syndrome_valid: bool
    correctable_estimate: int
    error_locator: list[int]
    error_positions: list[int]
    bits_flipped: int
    codeword_syndrome_valid: bool
    solver: str
    solver_work_units: int
    failure_reason: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """转换为普通字典，便于写 CSV / JSON。"""
        return asdict(self)

    @property
    def all_syndromes_zero(self) -> bool:
        """输入接收字的 Syndrome 是否全为 0。"""
        return not any(self.syndromes)


class BCHCode:
    """窄义原始二进制 BCH 码。

    参数
    ----
    m:
        扩域次数，域为 GF(2^m)。
    primitive_poly:
        本原多项式系数（低次优先，必须严格为 0/1 整数）。
    t:
        设计纠错能力（保证纠正任意 ``<= t`` 个错误）。
    """

    def __init__(self, m: int, primitive_poly: Sequence[int], t: int):
        if isinstance(t, bool) or not isinstance(t, int):
            raise TypeError("t 必须是整数。")
        if t < 1:
            raise ValueError("t 必须至少为 1。")
        self.field = GF2m(m, primitive_poly)
        self.m = self.field.m
        self.n = self.field.order  # 原始 BCH 码长度 = 2^m - 1
        self.t = int(t)
        if 2 * self.t > self.n - 1:
            raise ValueError(
                f"t={self.t} 过大：2t={2 * self.t} 必须不超过 n-1={self.n - 1}。"
            )

        self.root_powers = list(range(1, 2 * self.t + 1))
        self.coset_representatives, self.coset_members = self._build_root_cosets()
        self.roots = sorted({power % self.n for power in self.coset_members})
        self.generator_poly = self._build_generator_polynomial()
        self.r = poly_degree(self.generator_poly)  # 校验位数
        self.k = self.n - self.r  # 信息位数
        if self.k <= 0:
            raise ValueError(f"参数组合无效：n={self.n}, r={self.r}, k={self.k}。")
        self.parity_indices = list(range(self.r))
        self.data_indices = list(range(self.r, self.n))
        self.design_distance = 2 * self.t + 1
        # Syndrome 预计算幂表的惰性缓存（首次调用 ``syndromes`` 时构建）。
        self._syndrome_rows_cache: list[list[int]] | None = None
        self._validate_construction()

    # ------------------------------------------------------------------
    # 构造与自检
    # ------------------------------------------------------------------
    def _build_root_cosets(self) -> tuple[list[int], list[int]]:
        """把 ``α^1..α^{2t}`` 按 2-分圆陪集分组，并去掉重复根。"""
        representatives: list[int] = []
        members: list[int] = []
        seen: set[int] = set()
        for power in self.root_powers:
            if power in seen:
                continue
            coset = self.field.cyclotomic_coset(power)
            representatives.append(power)
            members.extend(coset)
            seen.update(coset)
        return representatives, sorted(members)

    def _build_generator_polynomial(self) -> list[int]:
        """``g(x) = ∏_{i ∈ 根集合}(x + α^i)``，系数必须全部落在 GF(2)。"""
        coefficients = [1]
        for power in self.roots:
            root = self.field.value_from_power(power)
            product = [0] * (len(coefficients) + 1)
            for index, coefficient in enumerate(coefficients):
                product[index] ^= self.field.mul(coefficient, root)
                product[index + 1] ^= coefficient
            coefficients = product
        for coefficient in coefficients:
            if coefficient not in (0, 1):
                raise RuntimeError(
                    f"生成多项式出现非二进制系数 {coefficient}，说明根集合或域运算有误。"
                )
        return trim(coefficients)

    def _validate_construction(self) -> None:
        """构造后逐条验证：次数、根条件、整除性、BCH 界。"""
        expected_degree = len(self.roots)
        if poly_degree(self.generator_poly) != expected_degree:
            raise RuntimeError(
                f"生成多项式次数 {poly_degree(self.generator_poly)} "
                f"与根个数 {expected_degree} 不一致。"
            )
        for power in self.root_powers:
            if self.field.poly_eval_power(self.generator_poly, power) != 0:
                raise RuntimeError(f"生成多项式在 α^{power} 处取值不为零。")
        x_pow_n_minus_1 = [1] + [0] * (self.n - 1) + [1]
        remainder = poly_mod(x_pow_n_minus_1, self.generator_poly)
        if poly_degree(remainder) >= 0:
            raise RuntimeError("生成多项式不能整除 x^n - 1。")
        if self.design_distance > self.n - self.k + 1:
            raise RuntimeError("BCH 界不成立，参数或根集合有误。")

    # ------------------------------------------------------------------
    # 输入校验
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_bits(bits: Sequence[int], expected: int, name: str) -> list[int]:
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

    def _validate_message(self, data_bits: Sequence[int]) -> list[int]:
        return self._validate_bits(data_bits, self.k, "data_bits")

    def _validate_codeword(self, codeword: Sequence[int]) -> list[int]:
        return self._validate_bits(codeword, self.n, "codeword")

    def _validate_positions(self, positions: Sequence[int] | None) -> list[int]:
        """严格校验"允许搜索/翻转的位置集合"。

        只接受**非布尔整数**下标：布尔值、浮点数、字符串一律 ``TypeError``，
        越界一律 ``ValueError``。这样缩短码接口不会因为 ``int()`` 转换或成员测试
        把 ``True``、``2.0`` 当作合法位置。
        """
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
    def encode(self, data_bits: Sequence[int]) -> list[int]:
        """系统编码：``c(x) = x^r·m(x) + (x^r·m(x) mod g(x))``（低次优先返回）。"""
        message = self._validate_message(data_bits)
        shifted = [0] * self.r + message
        remainder = poly_mod(shifted, self.generator_poly)
        codeword = shifted[:]
        for index, coefficient in enumerate(remainder):
            codeword[index] ^= coefficient
        return codeword

    def extract_data(self, codeword: Sequence[int]) -> list[int]:
        """从码字中取出信息位（下标 ``r..n-1``）。"""
        bits = self._validate_codeword(codeword)
        return [bits[index] for index in self.data_indices]

    # ------------------------------------------------------------------
    # Syndrome
    # ------------------------------------------------------------------
    def syndromes(self, received: Sequence[int]) -> list[int]:
        """计算 ``S_j = r(α^j)``，``j = 1..2t``（下标 0 对应 ``S_1``）。

        实现：**按位收集 + 预计算幂表异或**（见 :meth:`_syndrome_power_rows`）。

        数学上与原 Horner 法完全等价：因为 ``r`` 的系数只取 0/1，

            ``S_j = r(α^j) = Σ_p r_p · α^{j·p}``

        二进制下"乘以 bit"就是"是否计入异或"，于是只需把 **非零位置**（平均约 n/2 个）
        对应的 ``α^{j·p}`` 异或起来，无需对每个位置都做一次域乘法。
        """
        bits = self._validate_codeword(received)
        rows = self._syndrome_power_rows()
        # 只遍历**非零位置**；对偶校验码字通常约有一半为 1。
        set_positions = [position for position, bit in enumerate(bits) if bit]
        return [_xor_gather(row, set_positions) for row in rows]

    def _syndrome_power_rows(self) -> list[list[int]]:
        """预计算 ``α^{j·p}`` 表：``rows[j-1][p] = α^{j·p}``（``p = 0..n-1``）。

        只依赖域与码长，因此**每个码对象只构建一次**并缓存；
        构建时不做域乘法，而是复用 ``α^i`` 的顺序表 + 下标步进：

            ``(α^j)^p = α^{(j·p) mod ord(α)}``

        其中 ``ord(α) = 2^m - 1``。这样构建成本约为 ``2t · n`` 次**整数下标运算**，
        远低于同样次数的域乘法。
        """
        cached = self._syndrome_rows_cache
        if cached is not None:
            return cached
        field = self.field
        order = field.alpha_order
        # α^0..α^{order-1} 的顺序表（不重复计算域幂）
        power_values = [field.value_from_power(index) for index in range(max(order, self.n))]
        rows: list[list[int]] = []
        for power in range(1, 2 * self.t + 1):
            row = [0] * self.n
            index = 0
            for position in range(self.n):
                row[position] = power_values[index]
                index += power
                if index >= order:
                    index -= order * (index // order)
            rows.append(row)
        self._syndrome_rows_cache = rows
        return rows

    def _eval_at_power(self, bits: Sequence[int], power: int) -> int:
        """Horner 法求 ``r(α^power)``。

        **保留作为参考实现**：``syndromes()`` 已改用预计算幂表 + 异或收集，
        本方法用于测试中的逐项对照（验证优化没有改变数值结果）。
        """
        alpha_power = self.field.value_from_power(power)
        result = 0
        for coefficient in reversed(bits):
            result = self.field.mul(result, alpha_power) ^ (coefficient & 1)
        return result

    # ------------------------------------------------------------------
    # 错误定位多项式：求解器 3（经典 Berlekamp–Massey）
    # ------------------------------------------------------------------
    def solve_error_locator_bm(
        self, syndromes: Sequence[int], positions: Sequence[int] | None = None
    ) -> LocatorSolveResult:
        """用**经典 Berlekamp–Massey** 求错误定位多项式 σ（真正的 BM 迭代）。

        本方法**不建立、也不求解任何线性方程组**，因此与 ``newton_search`` /
        ``pgz`` 是**算法上独立**的第三条路径（BM 为 LFSR 综合）。

        数学设定
        --------
        以 ``a_n = S_{n+1}``（``n = 0..2t-1``）作为待综合序列，则第 n 步的偏差

            ``Δ_n = Σ_{i=0}^{L} C_i · S_{n+1-i}``

        恰好对应 Newton 恒等式的 ``k = n+1``。BM 求出的连接多项式
        ``C(x) = 1 + c_1 x + … + c_L x^L`` 就是 σ(x)（本项目 σ 的常数项恒为 1，
        与 ``σ(x) = ∏(1 + X_l x)`` 的约定一致）。

        递推（**带 b 归一化**，这是关键）
        --------------------------------
        .. code-block:: text

            C = 1, B = 1, L = 0, m = 1, b = 1
            for n in 0..2t-1:
                Δ = Σ_{i=0}^{deg C} C_i · a_{n-i}
                if Δ == 0:  m += 1; continue
                coef = Δ / b              # ← 必须除以**上一个非零偏差** b
                C = C + coef · x^m · B
                if 2L <= n:
                    L = n + 1 - L
                    B = old_C
                    b = Δ
                    m = 1
                else:
                    m += 1

        **为什么必须除以 b**：我此前多次实现失败，根因就是直接用 ``Δ`` 去乘辅助多项式
        ``B``，而没有归一化到"与上一次更新同一比例尺度"。少了 ``Δ/b`` 时，
        权重 1 的错误能过（因为 b 一开始就是 1），权重 ≥2 立刻错。

        接受条件（与另两条路径保持同一套语义）
        --------------------------------------
        1. ``L = deg C ≤ t``；
        2. 在允许位置上 Chien 搜索得到的根数恰好等于 ``deg C``。
        两者都满足才返回 ``solved=True``；超过纠错能力时如实返回未收敛。

        ``positions`` 用于缩短码：只在这些实际传输位置上做根数自洽校验。

        ``work_units`` 的口径：主循环中**发生多项式更新**（Δ≠0）的次数。
        """
        field = self.field
        sequence = [field._check_element(value) for value in syndromes]
        n_syms = len(sequence)
        search_positions = self._validate_positions(positions)

        # C、B 的系数列表低次优先，C[0] = 1 始终不变（保持 σ 的归一化形式）。
        C: list[int] = [1]
        B: list[int] = [1]
        length = 0
        shift = 1
        b = 1
        updates = 0

        for n in range(n_syms):
            # 偏差 Δ = Σ_{i=0}^{min(deg C, n)} C_i · a_{n-i}
            delta = 0
            for index in range(min(len(C) - 1, n) + 1):
                coefficient = C[index]
                if coefficient:
                    delta ^= field.mul(coefficient, sequence[n - index])
            if delta == 0:
                shift += 1
                continue

            updates += 1
            coef = field.div(delta, b)
            previous_C = C[:]
            size = max(len(C), len(B) + shift)
            updated = C + [0] * (size - len(C))
            for index, coefficient in enumerate(B):
                if coefficient:
                    updated[index + shift] ^= field.mul(coef, coefficient)
            C = trim(updated)

            if 2 * length <= n:
                length = n + 1 - length
                B = previous_C
                b = delta
                shift = 1
            else:
                shift += 1

        degree = poly_degree(C)
        if degree > self.t:
            return LocatorSolveResult(
                solver=SOLVER_BM,
                error_locator=C,
                error_count=degree,
                solved=False,
                work_units=updates,
                equations_used=0,
                detail=(
                    f"BM 得到 deg σ={degree} > t={self.t}，错误个数超出设计能力"
                ),
            )
        found, _ = self.chien_search(C, search_positions)
        if len(found) != degree:
            return LocatorSolveResult(
                solver=SOLVER_BM,
                error_locator=C,
                error_count=degree,
                solved=False,
                work_units=updates,
                equations_used=0,
                detail=(
                    f"BM 得到 deg σ={degree}，但在允许范围内只找到 {len(found)} 个根"
                ),
            )
        return LocatorSolveResult(
            solver=SOLVER_BM,
            error_locator=C,
            error_count=degree,
            solved=True,
            work_units=updates,
            equations_used=0,
            detail=f"BM 收敛：deg σ={degree}，更新 {updates} 次",
            # 已经完整搜过根，交给解码器复用，避免重复 Chien 搜索。
            error_positions=found,
        )

    # ------------------------------------------------------------------
    # 错误定位多项式：求解器 1（Newton 方程搜索，默认）
    # ------------------------------------------------------------------
    def solve_error_locator_newton_search(
        self, syndromes: Sequence[int], positions: Sequence[int] | None = None
    ) -> LocatorSolveResult:
        """求错误定位多项式 σ：**按错误个数递增筛选 Newton 方程**（默认解码器使用）。

        命名说明：本方法**不是** Berlekamp–Massey 迭代，因此不以 BM 命名。
        它是"逐次尝试错误个数、解 Newton 恒等式线性方程组、再做三重自洽校验"。

        参数
        ----
        positions:
            "允许出错的位置集合"，用于第 3 步的根数自洽校验。缩短码必须传入
            **实际传输的位置**，不得包含被固定为零的位置；``None`` 表示 ``0..n-1``。

        算法：

        1. ``deg_try = 1, 2, ..., t``：以 ``σ_1..σ_{deg_try}`` 为未知量、
           用 Newton 恒等式 ``k = deg_try+1 .. 2t`` 建正常方程，高斯消元求规范化解；
        2. 解必须**恰好**是 ``deg_try`` 次（前导系数非零），且**剩余全部** Newton
           恒等式都成立；
        3. 还要满足 ``deg σ = Chien 搜索找到的根数``（三重自洽）；
        4. 第一个通过全部检查的规模即为错误个数。

        为什么不做"经典 BM 迭代"
        ------------------------
        经典 BM 的迭代形式（偏差公式、辅助多项式用法、``m``/``L`` 更新规则）存在多个
        彼此等价但极易写混的变体：写错时会得到"次数看似合理、根位置却错误"的 σ
        （表现为"报告已纠正、输出满足全部 Syndrome，却纠到了错误的位"）。
        本项目在 3.1 阶段因此绕了很长弯路。本实现只依赖
        **已在测试中逐位验证** 的 Newton 恒等式 + 三重自洽校验，因此不会静默出错；
        代价是复杂度高于经典 BM 的 ``O(t^2)``，但在本阶段可接受，
        且长码会实测运行时间后如实报告。

        与 :meth:`solve_error_locator_pgz` 的关系：**两者共享同一类 Newton 方程与
        同一个高斯消元模块**，区别只在搜索顺序（本方法递增、PGZ 递减）与接受条件。
        因此它们的一致性检查属于"实现内部互检"，**不等同于跨算法的独立交叉验证**。

        ``work_units`` 的口径：本次尝试过的规模个数。
        """
        syndrome_list = [self.field._check_element(value) for value in syndromes]
        search_positions = self._validate_positions(positions)
        work_units = 0

        # 预先把"前 v 个方程"需要的右端量与系数按需切片，避免每轮重复构造。
        for degree_try in range(1, self.t + 1):
            work_units += 1
            locator, degree, status, equations = self._solve_square_newton_system(
                syndrome_list, degree_try
            )
            if status == SOLUTION_INCONSISTENT or degree != degree_try:
                # 该规模无一致解，或解退化（前导系数为零）→ 继续增大规模。
                continue
            if not self._newton_identities_hold(locator, syndrome_list):
                continue
            # 次数、根数、Newton 恒等式三者自洽，才认为求到了真正的 σ。
            found, _ = self.chien_search(locator, search_positions)
            if len(found) == degree:
                return LocatorSolveResult(
                    solver=SOLVER_NEWTON,
                    error_locator=locator,
                    error_count=degree,
                    solved=True,
                    work_units=work_units,
                    equations_used=equations,
                    detail=f"规模 {degree_try} 时自洽",
                    # 已经完整搜过根，交给解码器复用。
                    error_positions=found,
                )

        return LocatorSolveResult(
            solver=SOLVER_NEWTON,
            error_locator=[1],
            error_count=0,
            solved=False,
            work_units=work_units,
            equations_used=0,
            detail="所有规模都无法得到「次数 = 根数」自洽的 σ",
        )

    # ------------------------------------------------------------------
    # 错误定位多项式：求解器 2（PGZ 参考实现）
    # ------------------------------------------------------------------
    def solve_error_locator_pgz(
        self, syndromes: Sequence[int]
    ) -> LocatorSolveResult:
        """用 Peterson–Gorenstein–Zierler 解法求 σ（参考实现）。

        对 ``v = t, t-1, ..., 1`` 逐个尝试：以 ``σ_1..σ_v`` 为未知量、
        ``k = v+1 .. 2t`` 为方程解线性方程组；第一个"有解且次数恰为 v"的 v 即为错误个数。

        作为**参考实现**保留：它与 :meth:`solve_error_locator_newton_search` 输入相同、
        结果必须一致。但两者**共用同一类 Newton 方程与同一个高斯消元模块**，
        只在搜索顺序（本方法递减）与接受条件上不同，
        因此这种一致性属于**实现内部互检**，不是跨算法的独立交叉验证。

        ``work_units`` 的口径：所有尝试过的方程组规模之和 ``Σ(2t - v)``。
        """
        syndrome_list = [self.field._check_element(value) for value in syndromes]
        two_t = len(syndrome_list)
        work_units = 0
        for v in range(self.t, 0, -1):
            work_units += max(0, two_t - v)
            locator, degree, status, equations = self._solve_normal_equations(
                syndrome_list, v
            )
            if status == SOLUTION_INCONSISTENT:
                continue
            if status == SOLUTION_UNIQUE and degree == v:
                return LocatorSolveResult(
                    solver=SOLVER_PGZ,
                    error_locator=locator,
                    error_count=degree,
                    solved=True,
                    work_units=work_units,
                    equations_used=equations,
                    detail=f"v={v} 唯一解",
                )
            if degree < v:
                return LocatorSolveResult(
                    solver=SOLVER_PGZ,
                    error_locator=locator,
                    error_count=degree,
                    solved=True,
                    work_units=work_units,
                    equations_used=equations,
                    detail=f"v={v} 的解退化到次数 {degree}",
                )
        return LocatorSolveResult(
            solver=SOLVER_PGZ,
            error_locator=[1],
            error_count=0,
            solved=False,
            work_units=work_units,
            equations_used=0,
            detail="所有 v 都无法得到一致解",
        )

    # ------------------------------------------------------------------
    # 求解器公共部分
    # ------------------------------------------------------------------
    def _solve_square_newton_system(
        self, syndrome_list: Sequence[int], v: int
    ) -> tuple[list[int], int, str, int]:
        """用**前 v 个** Newton 恒等式解 ``σ_1..σ_v``（v×v 方阵，而不是超定系统）。

        数学依据：真实的 σ 满足**全部** Newton 恒等式，因此取其中任意 v 个
        （这里取 ``k = v+1 .. 2v``）都构成一致的方程组，且解唯一。
        与 :meth:`_solve_normal_equations`（用全部 ``2t - v`` 个方程）相比：

        * 结果相同（对 ``≤ t`` 个错误的情形）；
        * 代价从 ``O(t^3)`` 的消元降到 ``O(v^2)`` 一次方阵求解，
          使 ``t = 38`` 的长码求解从秒级降到毫秒级。

        安全性：解出候选后由调用方用 ``_newton_identities_hold`` 复核**全部**恒等式，
        因此"只取前 v 个方程"不会降低可靠性，只影响超能力情形下的候选选择。
        """
        if v <= 0:
            return [1], 0, SOLUTION_UNIQUE, 0
        # σ_0·S_k + σ_1·S_{k-1} + ... + σ_v·S_{k-v} = 0，取 k = v+1 .. 2v
        rows: list[list[int]] = []
        rhs: list[int] = []
        for k in range(v + 1, 2 * v + 1):
            if k > len(syndrome_list):
                break
            rows.append(
                [
                    syndrome_list[k - 1 - i] if k - 1 - i >= 0 else 0
                    for i in range(1, v + 1)
                ]
            )
            rhs.append(syndrome_list[k - 1])  # σ_0 = 1 移到右端
        if not rows:
            return [1], 0, SOLUTION_INCONSISTENT, 0
        result = solve_linear_system(self.field, rows, rhs)
        if result.status == SOLUTION_INCONSISTENT:
            return [1], 0, SOLUTION_INCONSISTENT, len(rows)
        solution = result.canonical_solution or [0] * v
        locator = trim([1] + [int(value) for value in solution])
        return locator, poly_degree(locator), result.status, len(rows)

    def _solve_normal_equations(
        self, syndrome_list: Sequence[int], v: int
    ) -> tuple[list[int], int, str, int]:
        """解 Newton 恒等式（未知量 ``σ_1..σ_v``），返回 ``(σ, 次数, 状态, 方程数)``。"""
        if v <= 0:
            return [1], 0, SOLUTION_UNIQUE, 0
        rows: list[list[int]] = []
        rhs: list[int] = []
        for k in range(v + 1, len(syndrome_list) + 1):  # k 是 1-based 的 Syndrome 下标
            rows.append(
                [
                    syndrome_list[k - 1 - i] if k - 1 - i >= 0 else 0
                    for i in range(1, v + 1)
                ]
            )
            rhs.append(syndrome_list[k - 1])  # σ_0 = 1 移到右端
        result = solve_linear_system(self.field, rows, rhs)
        if result.status == SOLUTION_INCONSISTENT:
            return [1], 0, SOLUTION_INCONSISTENT, len(rows)
        solution = result.canonical_solution or [0] * v
        locator = trim([1] + [int(value) for value in solution])
        return locator, poly_degree(locator), result.status, len(rows)

    def _newton_identities_hold(
        self, locator: Sequence[int], syndrome_list: Sequence[int]
    ) -> bool:
        """检查 ``σ_0·S_k + ... + σ_v·S_{k-v} = 0`` 对全部 ``k > deg σ`` 是否成立。"""
        degree = poly_degree(locator)
        for k in range(degree + 1, len(syndrome_list) + 1):
            total = syndrome_list[k - 1]
            for i in range(1, len(locator)):
                if locator[i] and k - 1 - i >= 0:
                    total ^= self.field.mul(locator[i], syndrome_list[k - 1 - i])
            if total != 0:
                return False
        return True

    def solve_error_locator(
        self,
        syndromes: Sequence[int],
        solver: str = SOLVER_NEWTON,
        positions: Sequence[int] | None = None,
    ) -> LocatorSolveResult:
        """按 ``solver`` 选择求解器（三条路径，命名与实现一致）。

        * ``newton_search``（默认）：按错误个数**递增**筛选 Newton 方程；
        * ``pgz``：按错误个数**递减**逐个解方程（与 Newton 共用同一数学基础）；
        * ``berlekamp_massey``：**经典 BM 迭代**，不建立也不求解线性方程组。

        ``positions`` 转发给支持该参数的求解器（缩短码用于限定允许出错的位置；
        PGZ 不接受该参数，其位置约束由随后的 Chien 搜索与根数检查施加）。
        """
        if solver == SOLVER_NEWTON:
            return self.solve_error_locator_newton_search(syndromes, positions=positions)
        if solver == SOLVER_BM:
            return self.solve_error_locator_bm(syndromes, positions=positions)
        if solver == SOLVER_PGZ:
            return self.solve_error_locator_pgz(syndromes)
        raise ValueError(
            f"未知求解器 {solver!r}；当前可用：{SOLVER_NEWTON!r}、{SOLVER_PGZ!r}、"
            f"{SOLVER_BM!r}。"
        )

    # ------------------------------------------------------------------
    # Chien Search
    # ------------------------------------------------------------------
    def chien_search(
        self, error_locator: Sequence[int], positions: Sequence[int] | None = None
    ) -> tuple[list[int], list[int]]:
        """在给定位置集合上搜索 σ 的根，返回 ``(错误位置列表, 根指数列表)``。

        * ``positions`` 默认 ``0..n-1``；缩短码必须传入"实际传输的位置"。
        * 位置 ``p`` 对应检验点 ``α^{-p}``；根指数记为 ``(-p) mod n``。
        """
        locator = [self.field._check_element(value) for value in error_locator]
        if poly_degree(locator) < 0:
            return [], []
        checked = self._validate_positions(positions)
        error_positions: list[int] = []
        root_exponents: list[int] = []
        for position in checked:
            exponent = (-position) % self.n
            if self.field.poly_eval(locator, self.field.value_from_power(exponent)) == 0:
                error_positions.append(position)
                root_exponents.append(exponent)
        return error_positions, root_exponents

    # ------------------------------------------------------------------
    # 解码
    # ------------------------------------------------------------------
    def decode(
        self,
        received: Sequence[int],
        positions: Sequence[int] | None = None,
        solver: str = SOLVER_NEWTON,
    ) -> DecodeResult:
        """BCH 解码：Syndrome → 求 σ → Chien → 位翻转 → **校验复核**。

        ``positions`` 用于缩短码：只允许在这些系数位置上定位并翻转错误。
        ``solver`` 选择错误定位求解器：``newton_search``（默认）、``pgz``（参考）、
        ``berlekamp_massey``（**经典 BM 迭代**，与另两条算法上独立）。
        """
        bits = self._validate_codeword(received)
        allowed = self._validate_positions(positions)
        syndromes = self.syndromes(bits)
        assert len(syndromes) == 2 * self.t

        if not any(syndromes):
            # 接收字已经落在码里。注意这**不代表没有发生错误**：
            # 它可能本身就是另一个合法码字，是否等于原始码字由调用方判定。
            # 也没有运行任何求解器 → ``solver`` 字段必须是 ``not_run``，
            # 不能把"请求的求解器"冒充成"实际执行的求解器"。
            return self._result(
                received=bits,
                corrected=bits[:],
                status=STATUS_NO_ERROR,
                syndromes=syndromes,
                locator=[1],
                error_positions=[],
                codeword_syndrome_valid=True,
                solver=NO_SOLVER_RUN,
                work_units=0,
                correctable_estimate=0,
                failure_reason="",
                notes=[
                    "Syndrome 全为 0，未做任何翻转，也未运行错误定位求解器"
                    f"（请求的求解器为 {solver!r}）。"
                ],
            )

        solved = self.solve_error_locator(syndromes, solver, positions=allowed)
        locator = solved.error_locator
        locator_degree = poly_degree(locator)
        # 复用求解器内部已完成的根搜索（newton/BM）；PGZ 内部不做根搜索时照旧自行搜索。
        if solved.error_positions is not None:
            error_positions = list(solved.error_positions)
        else:
            error_positions, _ = self.chien_search(locator, allowed)

        failure_reason = ""
        if not solved.solved:
            failure_reason = f"错误定位多项式求解未收敛：{solved.detail}"
        elif locator_degree < 0:
            failure_reason = "错误定位多项式为零多项式。"
        elif locator_degree > self.t:
            failure_reason = (
                f"错误定位多项式次数 {locator_degree} 超过纠错能力 t={self.t}，"
                "错误个数超出设计能力。"
            )
        elif locator_degree != len(error_positions):
            failure_reason = (
                f"错误定位多项式次数 {locator_degree} 与在允许范围内找到的根数 "
                f"{len(error_positions)} 不一致，无法得到一致的错误位置集合。"
            )
        elif not error_positions:
            failure_reason = "错误定位多项式在允许的位置范围内没有根。"

        if failure_reason:
            # 宁可不纠正，也不要基于不一致的错误位置集合乱翻。
            return self._result(
                received=bits,
                corrected=bits[:],
                status=STATUS_DECODE_FAILURE,
                syndromes=syndromes,
                locator=locator,
                error_positions=[],
                codeword_syndrome_valid=False,
                solver=solved.solver,
                work_units=solved.work_units,
                correctable_estimate=locator_degree,
                failure_reason=failure_reason,
            )

        corrected = bits[:]
        for position in error_positions:
            corrected[position] ^= 1
        recomputed = self.syndromes(corrected)
        if any(recomputed):
            # 复核失败：输出码字不满足校验关系，**不得**报告普通的 corrected。
            # 仍然返回候选码字供分析，但状态明确标记为复核失败。
            return self._result(
                received=bits,
                corrected=corrected,
                status=STATUS_CORRECTION_CHECK_FAILED,
                syndromes=syndromes,
                locator=locator,
                error_positions=error_positions,
                codeword_syndrome_valid=False,
                solver=solved.solver,
                work_units=solved.work_units,
                correctable_estimate=locator_degree,
                failure_reason="翻转后重新计算的 Syndrome 仍不为零。",
                notes=[f"复核失败后的 Syndrome：{recomputed}"],
            )

        return self._result(
            received=bits,
            corrected=corrected,
            status=STATUS_CORRECTED,
            syndromes=syndromes,
            locator=locator,
            error_positions=error_positions,
            codeword_syndrome_valid=True,
            solver=solved.solver,
            work_units=solved.work_units,
            correctable_estimate=locator_degree,
            failure_reason="",
        )

    def _result(
        self,
        *,
        received: list[int],
        corrected: list[int],
        status: str,
        syndromes: list[int],
        locator: list[int],
        error_positions: list[int],
        codeword_syndrome_valid: bool,
        solver: str,
        work_units: int,
        correctable_estimate: int,
        failure_reason: str,
        notes: list[str] | None = None,
    ) -> DecodeResult:
        return DecodeResult(
            received=received,
            corrected_codeword=corrected,
            decoded_data=self.extract_data(corrected),
            status=status,
            syndromes=list(syndromes),
            syndrome_valid=not any(syndromes),
            correctable_estimate=correctable_estimate,
            error_locator=list(locator),
            error_positions=list(error_positions),
            bits_flipped=len(error_positions),
            codeword_syndrome_valid=codeword_syndrome_valid,
            solver=solver,
            solver_work_units=work_units,
            failure_reason=failure_reason,
            notes=list(notes or []),
        )

    # ------------------------------------------------------------------
    # 分类（必须传入原始数据）
    # ------------------------------------------------------------------
    def classify(self, original_data: Sequence[int], result: DecodeResult) -> str:
        """把解码结果与**原始数据**比较后分类。

        * ``correctly_recovered``：输出数据等于原始数据，且解码器做过翻转；
        * ``no_error_no_change``：Syndrome 全为 0、无翻转，数据也正确；
        * ``miscorrected_wrong_data``：做过翻转且输出满足校验，但数据是错的（**误纠正**）；
        * ``correction_check_failed_wrong_data``：翻转后复核未通过且数据是错的；
        * ``decode_failure_not_recovered``：报告解码失败且数据未恢复；
        * ``valid_codeword_wrong_data``：Syndrome 全为 0 但数据错（接收字是另一个合法码字）。
        """
        original = self._validate_message(original_data)
        recovered = result.decoded_data == original
        if recovered:
            if result.status == STATUS_NO_ERROR:
                return CLASS_NO_ERROR
            return CLASS_CORRECTED
        if result.status == STATUS_DECODE_FAILURE:
            return CLASS_DECODE_FAILURE
        if result.status == STATUS_CORRECTION_CHECK_FAILED:
            return CLASS_VERIFY_FAILED
        if result.bits_flipped > 0:
            return CLASS_MISCORRECTED
        return CLASS_VALID_WRONG

    # ------------------------------------------------------------------
    # 参数与描述
    # ------------------------------------------------------------------
    def parameters(self) -> dict[str, Any]:
        """返回适合写入 CSV 的码参数（生成多项式由程序构造）。"""
        return {
            "m": self.m,
            "primitive_poly": poly_to_string(self.field.primitive_poly),
            "primitive_poly_coefficients_low_first": list(self.field.primitive_poly),
            "n": self.n,
            "k": self.k,
            "t": self.t,
            "r": self.r,
            "code_rate": self.k / self.n,
            "design_distance": self.design_distance,
            "root_powers": self.root_powers,
            "coset_representatives": self.coset_representatives,
            "union_of_cosets": self.roots,
            "generator_degree": poly_degree(self.generator_poly),
            "generator_poly_low_first": list(self.generator_poly),
            "generator_poly_string": poly_to_string(self.generator_poly),
            "syndrome_count": 2 * self.t,
            "parity_indices_low_first": self.parity_indices,
            "data_index_range_low_first": [self.r, self.n - 1],
        }

    def describe(self) -> str:
        return (
            f"BCH({self.n},{self.k}) t={self.t} 于 GF(2^{self.m})，"
            f"p(x) = {poly_to_string(self.field.primitive_poly)}，"
            f"g(x) = {poly_to_string(self.generator_poly)}，"
            f"校验位 r={self.r}，码率 {self.k / self.n:.4%}"
        )

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"BCHCode(m={self.m}, t={self.t}, n={self.n}, k={self.k}, r={self.r})"


def _xor_gather(row: Sequence[int], positions: Sequence[int]) -> int:
    """把 ``row`` 在给定位置上的取值全部异或起来（Syndrome 的热循环）。

    依据：``S_j = Σ_p r_p · α^{j·p}``，而 ``r_p ∈ {0,1}``，在 GF(2^m) 中
    "乘以 0/1"就是"是否计入异或"，因此只需对**非零位置**做异或，
    不必对每个位置都执行一次域乘法。
    """
    acc = 0
    for position in positions:
        acc ^= row[position]
    return acc


def bch_31_21() -> BCHCode:
    """第三阶段 3.1 指定的小码之一：BCH(31,21)，``t = 2``。"""
    return BCHCode(5, [1, 0, 1, 0, 0, 1], 2)


def bch_31_16() -> BCHCode:
    """第三阶段 3.1 指定的小码之二：BCH(31,16)，``t = 3``。"""
    return BCHCode(5, [1, 0, 1, 0, 0, 1], 3)


if __name__ == "__main__":  # pragma: no cover - 手工检查用
    for code in (bch_31_21(), bch_31_16()):
        print(code.describe())
        print(f"  根集合（α 的指数）= {code.roots}")
        print(f"  分圆陪集代表元 = {code.coset_representatives}")
        print(f"  校验位下标 = {code.parity_indices}")
        print(f"  数据位下标 = {code.data_indices[0]}..{code.data_indices[-1]}")
