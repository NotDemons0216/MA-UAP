"""GF(2^m) 有限域算术：加、乘、求逆、除法、指数/对数表、最小多项式。

数学约定
--------
1. 域元素有两种等价表示，本模块同时维护并在接口上明确区分：

   * **值表示（value）**：把系数按"低次项优先"打包成的整数。
     ``value = sum(c_i << i)``，即第 ``i`` 个二进制位对应多项式 ``x^i`` 的系数。
     例如 ``0b10110`` 表示 ``x^4 + x^2 + x``。
   * **幂表示（power）**：把元素写成 ``α^power``，``power`` 取值 ``0..order-1``；
     零元素单独用 ``None`` 表示（零没有对数）。

2. ``alpha`` 是满足本原多项式 ``p(α) = 0`` 的本原元。因此
   ``α^m = (p(x) - x^m) 在 α 处的取值``，对 ``p(x) = x^m + x^2 + 1`` 即 ``α^m = α^2 + 1``。
   程序不硬编码这个关系，而是"乘 2 后若溢出就异或掉 p(x) 的常数部分"来递推。

3. 有限域加法与减法都是按位异或（特征 2）。

4. 本模块用**值表示**做加法和乘 2（移位），用**对数/反对数表**做乘、除、求逆，
   两者的一致性由 ``test_gf2m.py`` 交叉验证（表查找结果必须等于朴素多项式乘法）。

5. 长度约定：所有"系数列表"统一使用**低次项优先**，
   即 ``poly[i]`` 是 ``x^i`` 的系数（与第三阶段任务书的位序约定一致）。

本模块只依赖 Python 标准库。
"""

from __future__ import annotations

from typing import Iterable, Sequence

# 校验/校验用的多项式系数只允许 0/1。
Bit = int


def _validate_positive_m(m: int) -> int:
    if isinstance(m, bool) or not isinstance(m, int):
        raise TypeError("m 必须是整数。")
    if m < 1:
        raise ValueError("m 必须至少为 1。")
    return int(m)


def poly_degree(coefficients: Sequence[int]) -> int:
    """返回多项式次数；零多项式返回 ``-1``（与数学惯例一致）。"""
    for index in range(len(coefficients) - 1, -1, -1):
        if coefficients[index]:
            return index
    return -1


def poly_to_string(coefficients: Sequence[int], variable: str = "x") -> str:
    """把低次优先的系数列表渲染成可读多项式字符串，便于日志与 README 引用。"""
    terms: list[str] = []
    for index in range(len(coefficients) - 1, -1, -1):
        if not coefficients[index]:
            continue
        if index == 0:
            terms.append("1")
        elif index == 1:
            terms.append(variable)
        else:
            terms.append(f"{variable}^{index}")
    return " + ".join(terms) if terms else "0"


def int_to_coefficients(value: int, length: int | None = None) -> list[int]:
    """整数（低次优先打包）→ 系数列表。``length`` 不足时左侧补零。"""
    if value < 0:
        raise ValueError("value 必须非负。")
    bits = [(value >> index) & 1 for index in range(value.bit_length())]
    if length is not None:
        if len(bits) > length:
            raise ValueError(f"value 需要 {len(bits)} 位，超过给定长度 {length}。")
        bits += [0] * (length - len(bits))
    return bits or [0]


def coefficients_to_int(coefficients: Sequence[int]) -> int:
    """系数列表 → 整数（低次优先打包）。"""
    value = 0
    for index, bit in enumerate(coefficients):
        if bit:
            value |= 1 << index
    return value


def trim(coefficients: Sequence[int]) -> list[int]:
    """去掉最高次侧的零系数。"""
    end = len(coefficients)
    while end > 1 and not coefficients[end - 1]:
        end -= 1
    return list(coefficients[:end])


def poly_mod(dividend: Sequence[int], divisor: Sequence[int]) -> list[int]:
    """低次优先系数列表上的 GF(2) 多项式取余。"""
    divisor = trim(divisor)
    if poly_degree(divisor) < 0:
        raise ZeroDivisionError("除数不能是零多项式。")
    remainder = trim(dividend)
    divisor_degree = poly_degree(divisor)
    while poly_degree(remainder) >= divisor_degree:
        shift = poly_degree(remainder) - divisor_degree
        for index in range(divisor_degree + 1):
            remainder[index + shift] ^= divisor[index]
        remainder = trim(remainder)
    if len(remainder) < divisor_degree:
        remainder += [0] * (divisor_degree - len(remainder))
    return remainder


def poly_mul(first: Sequence[int], second: Sequence[int]) -> list[int]:
    """低次优先系数列表上的 GF(2) 多项式乘法。"""
    if poly_degree(first) < 0 or poly_degree(second) < 0:
        return [0]
    product = [0] * (len(first) + len(second) - 1)
    for i, left in enumerate(first):
        if not left:
            continue
        for j, right in enumerate(second):
            if right:
                product[i + j] ^= 1
    return trim(product)


def poly_divmod(
    dividend: Sequence[int], divisor: Sequence[int]
) -> tuple[list[int], list[int]]:
    """GF(2) 多项式带余除法，返回 ``(商, 余数)``，均为低次优先。"""
    divisor = trim(divisor)
    divisor_degree = poly_degree(divisor)
    if divisor_degree < 0:
        raise ZeroDivisionError("除数不能是零多项式。")
    quotient = [0] * max(1, len(dividend))
    remainder = trim(dividend)
    while poly_degree(remainder) >= divisor_degree:
        shift = poly_degree(remainder) - divisor_degree
        if shift >= len(quotient):
            quotient += [0] * (shift + 1 - len(quotient))
        quotient[shift] ^= 1
        for index in range(divisor_degree + 1):
            remainder[index + shift] ^= divisor[index]
        remainder = trim(remainder)
    return trim(quotient), poly_mod(dividend, divisor)


class GF2m:
    """GF(2^m) 有限域，由本原多项式 ``p(x)`` 定义。

    参数
    ----
    m:
        扩域次数，必须是正整数。
    primitive_poly:
        本原多项式的系数列表（**低次优先**，且最高次系数必须为 1）。
        例如 ``x^5 + x^2 + 1`` 写成 ``[1, 0, 1, 0, 0, 1]``。
    """

    def __init__(self, m: int, primitive_poly: Sequence[int]):
        self.m = _validate_positive_m(m)
        # 系数必须**严格**是 0 或 1 的整数。不得写成 `int(bit) & 1` 之类的转换：
        # 那会把误传的 2、-1、1.5、None（部分情况）静默变成合法系数，
        # 使"用户写错了多项式"变成"域悄悄换成另一个"。测试里对每种非法输入都有断言。
        poly: list[int] = []
        for index, raw in enumerate(primitive_poly):
            if isinstance(raw, bool):
                poly.append(int(raw))
                continue
            if not isinstance(raw, int):
                raise TypeError(
                    f"本原多项式系数 primitive_poly[{index}]={raw!r} 必须是整数 0 或 1。"
                )
            if raw not in (0, 1):
                raise ValueError(
                    f"本原多项式系数 primitive_poly[{index}]={raw} 必须是 0 或 1"
                    "（不允许用其它整数或浮点数代替）。"
                )
            poly.append(raw)
        if poly_degree(poly) != self.m:
            raise ValueError(
                f"本原多项式次数必须是 m={self.m}，实际次数为 {poly_degree(poly)}。"
            )
        if poly[self.m] != 1:
            raise ValueError("本原多项式的最高次系数必须为 1。")
        if poly_degree(poly) >= 2 and poly[0] != 1:
            # x^m + ... + 1 形式：常数项为 1 才能保证 α 不为 0 的因子。
            raise ValueError("本原多项式的常数项必须为 1。")
        self.primitive_poly = poly

        self.order = (1 << self.m) - 1  # 非零元素个数 = 乘法群阶
        # 域元素上界（= 2^m），供 _check_element 的快速路径与完整校验共用。
        self._element_limit = 1 << self.m
        self._build_tables()

        if self.alpha_order != self.order:
            raise ValueError(
                f"给定多项式不是本原多项式：α 的阶为 {self.alpha_order}，"
                f"应为 {self.order}。"
            )

    # ------------------------------------------------------------------
    # 表构造
    # ------------------------------------------------------------------
    def _build_tables(self) -> None:
        """构造指数表与对数表，并同时记录 α 的阶。"""
        size = 1 << self.m
        # 约化多项式（去掉最高次项）打包成整数：乘法溢出时用它把 x^m 归约掉。
        reduction = coefficients_to_int(self.primitive_poly[:-1])
        antilog: list[int] = [0] * (2 * self.order)
        log: list[int | None] = [None] * size

        value = 1
        for power in range(self.order):
            if log[value] is not None:
                # 出现重复值说明 α 的阶小于域阶，会在构造后被拒绝。
                self.alpha_order = power
                self._antilog = antilog
                self._log = log
                return
            log[value] = power
            antilog[power] = value
            # 乘 α：整体左移一次（低次优先打包，位 i 对应 x^i），溢出则异或约化多项式。
            value <<= 1
            if value & size:
                value ^= reduction | size
        self.alpha_order = self.order
        if antilog[self.order - 1] == 0:
            raise RuntimeError("指数表构造不完整。")
        # 便于按 power 直接取模访问。
        for power in range(self.order, 2 * self.order):
            antilog[power] = antilog[power - self.order]
        self._antilog = antilog
        self._log = log

    # ------------------------------------------------------------------
    # 值 ↔ 幂
    # ------------------------------------------------------------------
    def value_from_power(self, power: int | None) -> int:
        """``α^power`` → 值表示；``None`` 表示零元素。"""
        if power is None:
            return 0
        if not isinstance(power, int):
            raise TypeError("power 必须是整数或 None。")
        return self._antilog[power % self.order]

    def power_from_value(self, value: int) -> int | None:
        """值表示 → ``α`` 的指数；零元素返回 ``None``。"""
        value = self._check_element(value)
        return self._log[value]

    def _check_element(self, value: int) -> int:
        """校验并返回一个域元素（位模式）。

        **内部快速路径**（性能优化，语义与原实现完全一致）：
        绝大多数内部运算传入的都是**普通 ``int`` 且已在合法范围内**的元素
        （例如查表结果、异或结果、指数表值），此时无需走 ``isinstance`` 分支——
        ``bool`` 的 ``__class__`` 是 ``bool``，不会被误放行。
        其余情况（``bool``/``float``/``str``/``None``/越界/负数）一律退回
        :meth:`_check_element_strict` 做完整校验，**拒绝行为不变**。
        """
        if value.__class__ is int and 0 <= value < self._element_limit:
            return value
        return self._check_element_strict(value)

    def _check_element_strict(self, value: int) -> int:
        """完整校验（对外部输入与公开接口保留，行为与优化前完全一致）。"""
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("域元素必须用整数表示。")
        if not 0 <= value < self._element_limit:
            raise ValueError(
                f"域元素必须落在 0..{self._element_limit - 1}，实际 {value}。"
            )
        return value

    def _check_elements(self, values: Iterable[int]) -> list[int]:
        return [self._check_element(value) for value in values]

    @staticmethod
    def _check_coefficient(value: int) -> int:
        """校验"系数"类型的数据。

        注意与 ``_check_element`` 的区别：**系数可以是 0/1 的普通整数**，
        不需要是域元素的位模式（例如多项式在整数下标下表示时可以出现大于 ``2^m-1``
        的下标值）。这里只要求非负整数，不做域范围检查。
        """
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"多项式系数必须是整数，实际 {value!r}。")
        if value < 0:
            raise ValueError(f"多项式系数必须非负，实际 {value}。")
        return value

    # ------------------------------------------------------------------
    # 基本运算
    # ------------------------------------------------------------------
    def add(self, a: int, b: int) -> int:
        """加法＝按位异或（特征 2）。"""
        return self._check_element(a) ^ self._check_element(b)

    def sub(self, a: int, b: int) -> int:
        """减法与加法相同。"""
        return self.add(a, b)

    def neg(self, a: int) -> int:
        """加法逆元就是自己。"""
        return self._check_element(a)

    def mul(self, a: int, b: int) -> int:
        """乘法：查表相加指数。零元素直接返回 0。"""
        a = self._check_element(a)
        b = self._check_element(b)
        if a == 0 or b == 0:
            return 0
        return self.value_from_power(self._log[a] + self._log[b])  # type: ignore[operator]

    def mul_power(self, a: int, power: int) -> int:
        """``a * α^power``，避免先算出 ``α^power`` 的值。"""
        a = self._check_element(a)
        if a == 0:
            return 0
        return self.value_from_power(self._log[a] + power)  # type: ignore[operator]

    def mul_naive(self, a: int, b: int) -> int:
        """朴素移位-异或乘法，用作 ``mul`` 的交叉验证（不用于热路径）。"""
        a = self._check_element(a)
        b = self._check_element(b)
        reduction = coefficients_to_int(self.primitive_poly[:-1])
        size = 1 << self.m
        product = 0
        while b:
            if b & 1:
                product ^= a
            b >>= 1
            a <<= 1
            if a & size:
                a ^= reduction | size
        return product

    def inv(self, a: int) -> int:
        """乘法逆元。零元素没有逆元，抛 ``ZeroDivisionError``。"""
        a = self._check_element(a)
        if a == 0:
            raise ZeroDivisionError("零元素没有乘法逆元。")
        return self.value_from_power(self.order - self._log[a])  # type: ignore[operator]

    def div(self, a: int, b: int) -> int:
        """除法 ``a / b``。"""
        b = self._check_element(b)
        if b == 0:
            raise ZeroDivisionError("除数为零。")
        a = self._check_element(a)
        if a == 0:
            return 0
        return self.value_from_power(self._log[a] - self._log[b])  # type: ignore[operator]

    def pow(self, a: int, exponent: int) -> int:
        """整数次幂；``exponent`` 可以为负，零的零次幂按 1 处理（与常见约定一致）。"""
        a = self._check_element(a)
        if exponent == 0:
            return 1
        if a == 0:
            if exponent < 0:
                raise ZeroDivisionError("零元素没有负次幂。")
            return 0
        return self.value_from_power(self._log[a] * exponent)  # type: ignore[operator]

    def square(self, a: int) -> int:
        return self.mul(a, a)

    # ------------------------------------------------------------------
    # 多项式求值
    # ------------------------------------------------------------------
    def poly_eval(self, coefficients: Sequence[int], x: int) -> int:
        """用 Horner 法求多项式在 ``x`` 处的取值（系数低次优先）。"""
        x = self._check_element(x)
        result = 0
        for coefficient in reversed(coefficients):
            result = self.mul(result, x) ^ self._check_coefficient(coefficient)
        return result

    def poly_eval_power(self, coefficients: Sequence[int], power: int | None) -> int:
        """求多项式在 ``α^power`` 处的取值。"""
        return self.poly_eval(coefficients, self.value_from_power(power))

    # ------------------------------------------------------------------
    # 分圆陪集与最小多项式
    # ------------------------------------------------------------------
    def cyclotomic_coset(self, representative: int) -> tuple[int, ...]:
        """返回包含 ``representative`` 的 2-分圆陪集（模 ``order``，按升序）。

        陪集定义为 ``{s, 2s, 4s, ...} mod order``；它是"同一个最小多项式的根"的集合。
        """
        if not isinstance(representative, int):
            raise TypeError("representative 必须是整数。")
        coset: set[int] = set()
        current = representative % self.order
        while current not in coset:
            coset.add(current)
            current = (current * 2) % self.order
        return tuple(sorted(coset))

    def minimal_polynomial(self, power: int) -> list[int]:
        """返回 ``α^power`` 在 GF(2) 上的最小多项式（系数低次优先）。

        构造方式：``m(x) = ∏_{i ∈ C(power)} (x - α^i) = ∏ (x + α^i)``，
        在 GF(2^m) 上展开后再验证所有系数都落在 GF(2)（即系数只能是 0 或 1）。
        不硬编码任何已知结果。
        """
        power = power % self.order
        roots = self.cyclotomic_coset(power)
        coefficients = [1]  # 从 1 开始逐个乘 (x + α^i)
        for root_power in roots:
            root = self.value_from_power(root_power)
            # (x + root)：低次优先 → [root, 1]
            product = [0] * (len(coefficients) + 1)
            for index, coefficient in enumerate(coefficients):
                product[index] ^= self.mul(coefficient, root)
                product[index + 1] ^= coefficient
            coefficients = product
        for coefficient in coefficients:
            if coefficient not in (0, 1):
                raise RuntimeError(
                    f"最简多项式系数 {coefficient} 不是 GF(2) 元素，"
                    f"说明分圆陪集或域运算有误（power={power}，陪集={roots}）。"
                )
        return trim(coefficients)

    # ------------------------------------------------------------------
    # 描述
    # ------------------------------------------------------------------
    def parameters(self) -> dict[str, object]:
        return {
            "m": self.m,
            "field_order": 1 << self.m,
            "multiplicative_order": self.order,
            "primitive_poly_degree": poly_degree(self.primitive_poly),
            "primitive_poly_coefficients_low_first": list(self.primitive_poly),
            "primitive_poly_string": poly_to_string(self.primitive_poly),
            "alpha_order": self.alpha_order,
        }

    def describe(self) -> str:
        return (
            f"GF(2^{self.m})，本原多项式 p(x) = {poly_to_string(self.primitive_poly)}，"
            f"非零元素 {self.order} 个，α 的阶 = {self.alpha_order}"
        )

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"GF2m(m={self.m}, primitive_poly={self.primitive_poly})"


def gf32() -> GF2m:
    """第三阶段 3.1 指定的域：GF(2^5)，``p(x) = x^5 + x^2 + 1``。"""
    return GF2m(5, [1, 0, 1, 0, 0, 1])


if __name__ == "__main__":  # pragma: no cover - 手工检查用
    field = gf32()
    print(field.describe())
    print("α 的前若干次幂：")
    for power in range(6):
        value = field.value_from_power(power)
        print(f"  α^{power:<2} = {value:>2} = 0b{value:05b}")
    for representative in (0, 1, 3, 5, 7):
        coset = field.cyclotomic_coset(representative)
        minimal = field.minimal_polynomial(representative)
        print(
            f"陪集 C{representative} = {list(coset)}，"
            f"最小多项式 = {poly_to_string(minimal)}（次数 {poly_degree(minimal)}）"
        )
