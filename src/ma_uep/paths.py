"""项目根目录的唯一定义。

布局约定（开发目录与发布仓库一致）：

    <PROJECT_ROOT>/
    ├── run.py
    ├── configs/
    ├── src/ma_uep/...
    └── data/
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 可供覆盖的默认目录
DEFAULT_MODEL_PATH = "models/Qwen2.5-1.5B-Base"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "v1_default.yaml"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
VENDORED_BCH = PROJECT_ROOT / "src" / "ma_uep" / "ecc" / "_vendor" / "BCH"


def resolve(path: str | Path) -> Path:
    """把配置中的相对路径解析为绝对路径（相对 PROJECT_ROOT）。"""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)
