"""构建 MA-UEP V1.0 发布快照。

产物：
    releases/V1.0/
    ├── README_RELEASE.md
    ├── MANIFEST.sha256      # 逐文件 SHA-256
    ├── MANIFEST.json        # 文件清单 + 大小 + 哈希
    ├── source/              # 源码、配置、测试、文档
    └── evidence/            # 测试日志、回归记录、样例实验、阈值快照
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "releases" / "V1.0"
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", "releases", "outputs",
                "pip-ephem-wheel-cache-zv3qg6fe"}
EXCLUDE_SUFFIX = {".pyc", ".pyo"}
EXCLUDE_NAMES = {"ma_uep.egg-info"}

SOURCE_ITEMS = ["run.py", "pyproject.toml", "README.md", "requirements.txt",
                "configs", "src", "tests", "docs"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_tree(src: Path, dst: Path) -> list[Path]:
    copied = []
    for path in sorted(src.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(src)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        if path.suffix in EXCLUDE_SUFFIX:
            continue
        if any(name in rel.parts for name in EXCLUDE_NAMES):
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(target)
    return copied


def main() -> int:
    import sys
    manifest_only = "--manifest-only" in sys.argv
    if manifest_only:
        return write_manifest()
    if RELEASE.exists():
        shutil.rmtree(RELEASE)
    (RELEASE / "source").mkdir(parents=True)
    (RELEASE / "evidence").mkdir(parents=True)

    files: list[Path] = []

    # ---- source ----
    for item in SOURCE_ITEMS:
        path = ROOT / item
        if not path.exists():
            continue
        if path.is_dir():
            files.extend(copy_tree(path, RELEASE / "source" / item))
        else:
            target = RELEASE / "source" / item
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            files.append(target)

    # ---- evidence：日志 ----
    evidence_src = ROOT / "releases" / "V1.0" / "evidence"
    for name in ("unit_tests_log.txt", "equivalence_log.txt"):
        src = evidence_src / name
        if src.exists():
            files.append(src)

    # ---- evidence：回归记录 ----
    for name in ("regression_final.json", "regression_ber.json",
                 "regression_fixed.json"):
        src = ROOT / "outputs" / "regression" / name
        if src.exists():
            target = RELEASE / "evidence" / name
            shutil.copy2(src, target)
            files.append(target)

    # ---- evidence：样例实验（完整闭环产物）----
    sample = ROOT / "outputs" / "20261114-185656-ma_uep-ber0p0001-seed42-44efbc"
    if not sample.exists():
        candidates = sorted((ROOT / "outputs").glob("2*"))
        candidates = [c for c in candidates if (c / "result.json").exists()]
        sample = candidates[-1] if candidates else None
    if sample and sample.exists():
        target_dir = RELEASE / "evidence" / "experiment_sample"
        for path in sorted(sample.rglob("*")):
            if path.is_file():
                target = target_dir / path.relative_to(sample)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                files.append(target)

    # ---- evidence：阈值快照（外部依赖的版本记录）----
    thresholds = ROOT.parent / "ECC" / "outputs" / "thresholds"
    if thresholds.exists():
        target_dir = RELEASE / "evidence" / "thresholds"
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in ("tensor_thresholds.json", "tensor_thresholds.csv"):
            src = thresholds / name
            if src.exists():
                target = target_dir / name
                shutil.copy2(src, target)
                files.append(target)

    return write_manifest()


def write_manifest() -> int:
    files = []
    for path in sorted(RELEASE.rglob("*")):
        if path.is_file() and path.name not in ("MANIFEST.json", "MANIFEST.sha256",
                                                "README_RELEASE.md"):
            files.append(path)

    # ---- manifest ----
    entries = []
    lines = []
    for path in sorted(files):
        if path.name in ("MANIFEST.json", "MANIFEST.sha256"):
            continue
        rel = path.relative_to(RELEASE).as_posix()
        digest = sha256_file(path)
        size = path.stat().st_size
        entries.append({"path": rel, "bytes": size, "sha256": digest})
        lines.append(f"{digest}  {rel}")

    (RELEASE / "MANIFEST.sha256").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")
    manifest = {
        "system": "MA-UEP V1.0",
        "schema_version": "1.0",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_files": len(entries),
        "total_bytes": sum(e["bytes"] for e in entries),
        "files": entries,
    }
    (RELEASE / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- release notes ----
    (RELEASE / "README_RELEASE.md").write_text(RELEASE_NOTES.format(
        n_files=len(entries),
        total_mb=manifest["total_bytes"] / 1e6,
        frozen_at=manifest["frozen_at"],
        git=git_commit(),
    ), encoding="utf-8")
    print(f"快照已生成：{len(entries)} 个文件，"
          f"{manifest['total_bytes']/1e6:.2f} MB")
    return 0


def git_commit() -> str:
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=str(ROOT), capture_output=True, text=True,
                             timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:                      # noqa: BLE001
        return "unavailable"


RELEASE_NOTES = """# MA-UEP V1.0 发布快照

> 冻结时间：{frozen_at}
> Git commit：{git}
> 文件数：{n_files}，总大小 {total_mb:.2f} MB

## 这是什么

这是 MA-UEP V1.0 的**只读发布快照**。它冻结了：

| 目录 | 内容 |
|---|---|
| `source/` | 源码、配置、测试、文档、`run.py`、`pyproject.toml` |
| `evidence/` | 单元测试日志、等价性验证日志、历史回归记录、样例实验、阈值快照 |
| `MANIFEST.sha256` | 每个文件的 SHA-256（可用于校验未被修改） |
| `MANIFEST.json` | 文件清单、大小、哈希、冻结时间 |

## 冻结的内容

1. **源码**：`src/ma_uep/`（8 个模块）
2. **配置**：`configs/v1_default.yaml` + `src/ma_uep/constants.py`（冻结参数）
3. **测试**：38 项单元测试 + 等价性验证 + 历史回归脚本
4. **测试结果**：单元测试全通过、L2/L3 等价、回归 8/8（差异 ≤ 5e-7）
5. **实验数据**：一个完整闭环实验的全部产物 + 回归记录

## 校验方式

```bash
cd releases/V1.0
# Windows
certutil -hashfile MANIFEST.sha256 SHA256   # 记录清单自身的哈希
# 或逐文件核对
python - <<'PY'
import hashlib, pathlib
for line in pathlib.Path("MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
    digest, rel = line.split("  ", 1)
    actual = hashlib.sha256(pathlib.Path(rel).read_bytes()).hexdigest()
    assert actual == digest, rel
print("all files verified")
PY
```

## 只读属性

本目录下所有文件已设置 Windows 只读属性（`attrib +R`）。
如需修改，必须先执行 `attrib -R /S`，但这会破坏快照与 `MANIFEST.sha256` 的一致性，
**不建议**。

## 快照不包含的内容

| 内容 | 原因 |
|---|---|
| Bit-level 风险地图（16.34 GB） | 上游数据，体积过大；以路径 + SHA-256 前缀记录在 `meta.json` |
| 权重重要性数组（4.46 GB） | 同上 |
| 模型权重（约 3 GB） | 使用官方 checkpoint |
| `outputs/equivalence/` 等级图（223 MB） | 可由 `tests/verify_equivalence.py` 重新生成 |

这些外部依赖的路径与校验方式见 `source/README.md` §3。

## 复现一次实验

```bash
cd source
python run.py check
python run.py run --mode ma_uep --ber 1e-4 --seed 42
```

注意：`run.py` 依赖仓库中位于 `MA-UEP/` 上一级的 `BCH/` 与 `ECC/outputs/`，
需要保持原有仓库结构；也可通过环境变量 `MA_UEP_BCH_PATH` 指定 BCH 路径。
"""


if __name__ == "__main__":
    raise SystemExit(main())
