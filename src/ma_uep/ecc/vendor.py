"""现有 BCH 母码的唯一接入点。

查找顺序：
  1. 环境变量 MA_UEP_BCH_PATH
  2. 显式设置（set_bch_path）
  3. 仓库内 vendored 副本：src/ma_uep/ecc/_vendor/BCH
  4. 开发布局回退：PROJECT_ROOT 的同级 BCH/

MA-UEP 不重新实现 BCH，只包装已有的母码实现。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ..paths import PROJECT_ROOT, VENDORED_BCH

_BCH_PATH: Path | None = None


def set_bch_path(path: str | Path | None) -> None:
    global _BCH_PATH
    _BCH_PATH = Path(path).resolve() if path else None
    if _BCH_PATH and str(_BCH_PATH) not in sys.path:
        sys.path.insert(0, str(_BCH_PATH))


def candidate_paths() -> list[Path]:
    paths = []
    env = os.environ.get("MA_UEP_BCH_PATH")
    if env:
        paths.append(Path(env).resolve())
    if _BCH_PATH is not None:
        paths.append(_BCH_PATH)
    paths.append(VENDORED_BCH)
    paths.append(PROJECT_ROOT.parent / "BCH")
    return paths


def bch_path() -> Path:
    for path in candidate_paths():
        if path.exists():
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            return path
    tried = "\n".join(f"  - {p}" for p in candidate_paths())
    raise FileNotFoundError(
        "找不到 BCH 母码目录。已尝试：\n" + tried +
        "\n请设置环境变量 MA_UEP_BCH_PATH 或在配置中指定 ecc.bch_path")


def import_bch_code():
    bch_path()
    from bch import BCHCode  # type: ignore
    return BCHCode
