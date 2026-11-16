#!/usr/bin/env python
"""MA-UEP V1.0 统一入口。

用法：
    python run.py --help
    python run.py config --show
    python run.py check
    python run.py run --mode ma_uep --ber 1e-4 --seed 42
    python run.py report
    python run.py test
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ma_uep.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
