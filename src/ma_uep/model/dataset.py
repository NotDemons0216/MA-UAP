"""WikiText-2 加载与评估窗口构造。

三级回退（解决审计中的 B4）：
  1. 配置显式指定的本地文件（txt 或 arrow）；
  2. `datasets` 库标准加载；
  3. 直接读取 HuggingFace 缓存的 arrow 文件。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..constants import DATASET_NAME, DATASET_SPLIT, EVAL_MAX_LENGTH, EVAL_STRIDE


@dataclass
class EvalSet:
    input_ids: torch.Tensor
    windows: list[tuple[int, int, int]]
    labels: list[torch.Tensor]
    source: str
    n_tokens: int


def _read_arrow_text(path: Path, split: str) -> str:
    import pyarrow as pa
    with pa.memory_map(str(path), "r") as source:
        table = pa.ipc.open_stream(source).read_all()
    return "\n\n".join(table.column("text").to_pylist())


def _find_cached_arrow(split: str) -> Path | None:
    cache = Path.home() / ".cache" / "huggingface" / "datasets" / "Salesforce___wikitext"
    if not cache.exists():
        return None
    for path in cache.rglob(f"wikitext-{split}.arrow"):
        return path
    return None


def load_text(split: str = DATASET_SPLIT, dataset_path: str | Path | None = None) -> tuple[str, str]:
    """返回 (文本, 来源描述)。"""
    if dataset_path:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"指定的数据集文件不存在: {path}")
        if path.suffix == ".arrow":
            return _read_arrow_text(path, split), f"arrow:{path}"
        return path.read_text(encoding="utf-8"), f"text:{path}"

    try:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", DATASET_NAME, split=split)
        return "\n\n".join(ds["text"]), "datasets:hub"
    except Exception as exc:                      # noqa: BLE001
        cached = _find_cached_arrow(split)
        if cached is None:
            raise RuntimeError(
                "无法加载 WikiText-2：datasets 失败且没有本地缓存。"
                "请在配置中指定 dataset.path") from exc
        return _read_arrow_text(cached, split), f"arrow_cache:{cached}"


def build_windows(tokenizer, text: str, max_length: int = EVAL_MAX_LENGTH,
                  stride: int = EVAL_STRIDE) -> EvalSet:
    import torch as _torch
    ids = tokenizer(text, add_special_tokens=False,
                    return_tensors="pt")["input_ids"]
    input_ids = ids.to("cuda:0" if _torch.cuda.is_available() else "cpu")
    windows, prev = [], 0
    for begin in range(0, input_ids.shape[1], stride):
        end = min(begin + max_length, input_ids.shape[1])
        target = end - prev
        if target <= 0:
            break
        windows.append((begin, end, target))
        prev = end
        if end >= input_ids.shape[1]:
            break
    labels = []
    for begin, end, target in windows:
        window = input_ids[:, begin:end]
        label = window.clone()
        if target < label.shape[1]:
            label[:, :-target] = -100
        labels.append(label)
    return EvalSet(input_ids=input_ids, windows=windows, labels=labels,
                   source="", n_tokens=int(input_ids.shape[1]))
