"""实验注册表：实验 ID、目录结构与元数据。"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

from ..constants import SCHEMA_VERSION, SYSTEM_VERSION
from ..paths import PROJECT_ROOT


def git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=repo_root, capture_output=True, text=True,
                             timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:                        # noqa: BLE001
        return "unavailable"


def environment() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "system_version": SYSTEM_VERSION,
        "schema_version": SCHEMA_VERSION,
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:                        # noqa: BLE001
        info["torch"] = "unavailable"
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:                        # noqa: BLE001
        info["transformers"] = "unavailable"
    return info


def new_experiment_id(mode: str, params: dict) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    parts = [stamp, mode]
    for key in ("ber", "k_bits", "seed"):
        if key in params and params[key] is not None:
            parts.append(f"{key}{params[key]}")
    parts.append(suffix)
    return "-".join(str(p) for p in parts).replace(".", "p")


class ExperimentRegistry:
    def __init__(self, root: str | Path, repo_root: Path | None = None):
        self.root = Path(root)
        self.repo_root = repo_root or PROJECT_ROOT
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, experiment_id: str) -> Path:
        path = self.root / experiment_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_meta(self, path: Path, config, extra: dict) -> dict:
        meta = {
            "experiment_id": path.name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": git_commit(self.repo_root),
            "environment": environment(),
            "config": config.to_dict(),
            **extra,
        }
        (path / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        (path / "config.snapshot.json").write_text(config.snapshot(),
                                                   encoding="utf-8")
        return meta

    def write_result(self, path: Path, payload: dict) -> None:
        payload = dict(payload)
        payload["schema_version"] = SCHEMA_VERSION
        (path / "result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")

    def list_experiments(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def load_results(self) -> list[dict]:
        results = []
        for path in sorted(self.root.iterdir()):
            result_file = path / "result.json"
            if result_file.exists():
                payload = json.loads(result_file.read_text(encoding="utf-8"))
                payload.setdefault("experiment_id", path.name)
                results.append(payload)
        return results
