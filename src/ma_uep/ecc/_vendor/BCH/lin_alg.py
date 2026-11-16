"""GF(2^m) 上的线性方程组求解（高斯消元）。

用途：BCH 解码中由 Newton 恒等式解出错误定位多项式的系数
（即 Peterson–Gorenstein–Zierler 解法）。这一步**只依赖已在测试中逐位验证过的
Newton 恒等式**，不涉及 Berlekamp–Massey 迭代中的符号约定，

    ``σ_0·S_k + σ_1·S_{k-1} + ... + σ_v·S_{k-v} = 0``（``k = v+1 .. 2t``，``σ_0 = 1``），

因此不会因为"偏差公式/辅助多项式用法"这类约定错误而给出错误的错误位置。

约定
----
* 所有域元素用 :class:`gf2m.GF2m` 的值表示（非负整数，位模式）。
* 加法为异或，消元时的行变换为 ``行 ← 行 + 因子·主元行``。
* 求解结果区分三种情况：唯一解、无解（不一致）、多解（自由变量）。
"""

from __future__ import annotations

from typing import Sequence

from gf2m import GF2m

# 解的状态
SOLUTION_UNIQUE = "unique"
SOLUTION_INCONSISTENT = "inconsistent"
SOLUTION_UNDERDETERMINED = "underdetermined"


class LinearSolveResult:
    """一次线性方程组求解的结果。"""

    __slots__ = (
        "status",
        "solution",
        "rank",
        "pivot_columns",
        "free_columns",
        "canonical_solution",
    )

    def __init__(
        self,
        status: str,
        solution: list[int] | None,
        rank: int,
        pivot_columns: list[int],
        free_columns: list[int],
        canonical_solution: list[int] | None = None,
    ):
        self.status = status
        self.solution = solution
        self.rank = rank
        self.pivot_columns = pivot_columns
        self.free_columns = free_columns
        # 多解时的规范化解（自由变量取 0）；唯一解时与 solution 相同。
        self.canonical_solution = (
            canonical_solution if canonical_solution is not None else solution
        )

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"LinearSolveResult(status={self.status}, rank={self.rank}, "
            f"pivots={self.pivot_columns}, free={self.free_columns})"
        )


def solve_linear_system(
    field: GF2m,
    matrix: Sequence[Sequence[int]],
    rhs: Sequence[int],
) -> LinearSolveResult:
    """在高斯消元下求解 ``A x = b``。

    参数
    ----
    matrix:
        行优先的系数矩阵（每行长度相同）。
    rhs:
        右端向量，长度必须等于行数。

    返回 :class:`LinearSolveResult`。零行且右端非零 ⇒ 不一致（无解）；
    存在自由变量 ⇒ 多解；否则唯一解。
    """
    rows = [list(row) for row in matrix]
    if len(rows) != len(rhs):
        raise ValueError(
            f"矩阵行数 {len(rows)} 与右端长度 {len(rhs)} 不一致。"
        )
    if not rows:
        return LinearSolveResult(SOLUTION_UNDERDETERMINED, None, 0, [], [])
    columns = len(rows[0])
    for index, row in enumerate(rows):
        if len(row) != columns:
            raise ValueError(f"第 {index} 行长度为 {len(row)}，与其他行不一致。")
    augmented = [
        [field._check_element(value) for value in row] + [field._check_element(rhs[index])]
        for index, row in enumerate(rows)
    ]

    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(columns):
        # 选主元：第一个该列非零的行。
        selected = None
        for candidate in range(pivot_row, len(augmented)):
            if augmented[candidate][column]:
                selected = candidate
                break
        if selected is None:
            continue
        augmented[pivot_row], augmented[selected] = (
            augmented[selected],
            augmented[pivot_row],
        )
        pivot_value = augmented[pivot_row][column]
        inverse = field.inv(pivot_value)
        # 主元归一化为 1（注意：只在域元素上做乘法，右端同样缩放）。
        augmented[pivot_row] = [
            field.mul(value, inverse) for value in augmented[pivot_row]
        ]
        # 其余行消元。
        for other in range(len(augmented)):
            if other == pivot_row:
                continue
            factor = augmented[other][column]
            if factor:
                augmented[other] = [
                    field.add(value, field.mul(factor, pivot_value_2))
                    for value, pivot_value_2 in zip(
                        augmented[other], augmented[pivot_row]
                    )
                ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == len(augmented):
            break

    # 检查不一致：零系数行但右端非零。
    for row in augmented:
        if all(value == 0 for value in row[:columns]) and row[columns] != 0:
            return LinearSolveResult(
                SOLUTION_INCONSISTENT, None, len(pivot_columns), pivot_columns, []
            )

    free_columns = [index for index in range(columns) if index not in pivot_columns]
    # 规范化解：主元变量取化简后的右端值，自由变量取 0。
    canonical = [0] * columns
    for row_index, column in enumerate(pivot_columns):
        canonical[column] = augmented[row_index][columns]
    if free_columns:
        return LinearSolveResult(
            SOLUTION_UNDERDETERMINED,
            None,
            len(pivot_columns),
            pivot_columns,
            free_columns,
            canonical_solution=canonical,
        )

    # 唯一解：所有列都是主元列。
    return LinearSolveResult(
        SOLUTION_UNIQUE,
        canonical,
        len(pivot_columns),
        pivot_columns,
        [],
        canonical_solution=canonical,
    )
