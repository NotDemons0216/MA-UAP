"""模型管理：加载、BF16 校验、目标 Tensor 枚举、权重备份与恢复。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from ..constants import BLOCKS, MODULES, MODEL_DTYPE, COVERAGE_NOTE
from ..paths import PROJECT_ROOT


@dataclass(frozen=True)
class TensorSpec:
    block_id: int
    module: str
    n_weights: int
    shape: tuple[int, int]

    @property
    def name(self) -> str:
        return f"L{self.block_id}.{self.module}"


def resolve_model_path(path: str | Path, repo_root: Path | None = None) -> Path:
    """解析模型路径。

    顺序：绝对路径 → 项目根下 → 项目根的上级（开发布局兼容）。
    """
    p = Path(path)
    if p.is_absolute():
        return p
    root = repo_root or PROJECT_ROOT
    for base in (root, root.parent):
        candidate = (base / p).resolve()
        if candidate.exists():
            return candidate
    return (root / p).resolve()


class ModelManager:
    """持有模型，并提供"取原始位模式 / 写回"的成对操作。"""

    def __init__(self, path: str | Path, device: str = "cuda:0",
                 repo_root: Path | None = None):
        self.path = resolve_model_path(path, repo_root)
        self.device = device
        self.model = None
        self._backup: dict[str, np.ndarray] = {}

    # ---------------- 加载 ----------------
    def load(self):
        """加载模型。

        优先用 accelerate 的 device_map 直接落到 GPU（CPU 峰值低，
        在内存紧张的机器上更稳）；失败则回退到 .to(device)。
        """
        from transformers import AutoModelForCausalLM
        import os
        use_device_map = os.environ.get("MA_UEP_DEVICE_MAP", "0") == "1"
        if use_device_map:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.path), dtype=torch.bfloat16, low_cpu_mem_usage=True,
                device_map=self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.path), dtype=torch.bfloat16, low_cpu_mem_usage=True)
            self.model = self.model.to(self.device)
        self.model = self.model.eval()
        self._check_bf16()
        return self

    def _check_bf16(self) -> None:
        bad = [n for n, p in self.model.named_parameters()
               if p.dtype not in (torch.bfloat16, torch.float32)]
        if bad:
            raise RuntimeError(f"非 BF16 参数: {bad[:5]}")

    # ---------------- 结构 ----------------
    def get_module(self, block_id: int, module: str):
        block = self.model.model.layers[block_id]
        if module in ("q_proj", "k_proj", "v_proj", "o_proj"):
            return getattr(block.self_attn, module)
        return getattr(block.mlp, module)

    def iter_target_tensors(self, blocks: Iterable[int] = BLOCKS,
                            modules: Iterable[str] = MODULES):
        for block_id in blocks:
            for module in modules:
                weight = self.get_module(block_id, module).weight
                yield TensorSpec(block_id=block_id, module=module,
                                 n_weights=int(weight.numel()),
                                 shape=(int(weight.shape[0]), int(weight.shape[1])))

    def describe(self) -> dict:
        return {
            "path": str(self.path),
            "dtype": MODEL_DTYPE,
            "device": self.device,
            "n_layers": len(self.model.model.layers),
            "coverage_note": COVERAGE_NOTE,
        }

    # ---------------- 权重读写 ----------------
    def read_raw(self, spec: TensorSpec) -> np.ndarray:
        weight = self.get_module(spec.block_id, spec.module).weight
        return weight.view(torch.int16).reshape(-1).cpu().numpy().view(np.uint16)

    def write_raw(self, spec: TensorSpec, raw_u16: np.ndarray) -> None:
        weight = self.get_module(spec.block_id, spec.module).weight
        with torch.no_grad():
            weight.view(torch.int16).reshape(-1).copy_(
                torch.from_numpy(raw_u16.astype(np.int16)).to(weight.device))

    def write_float(self, spec: TensorSpec, values: np.ndarray) -> None:
        weight = self.get_module(spec.block_id, spec.module).weight
        tensor = torch.from_numpy(values.astype(np.float32)).to(torch.bfloat16)
        with torch.no_grad():
            weight.copy_(tensor.reshape(weight.shape).to(weight.device))

    # ---------------- 备份 / 恢复 ----------------
    def backup(self, specs: Iterable[TensorSpec] | None = None) -> int:
        specs = list(specs) if specs is not None else \
            list(self.iter_target_tensors())
        for spec in specs:
            self._backup[spec.name] = self.read_raw(spec).copy()
        return len(self._backup)

    def restore(self) -> int:
        restored = 0
        for spec in self.iter_target_tensors():
            if spec.name in self._backup:
                self.write_raw(spec, self._backup[spec.name])
                restored += 1
        return restored

    def weight_fingerprint(self, specs: Iterable[TensorSpec] | None = None) -> str:
        specs = list(specs) if specs is not None else list(self.iter_target_tensors())
        digest = hashlib.sha256()
        for spec in specs:
            digest.update(self.read_raw(spec).tobytes())
        return digest.hexdigest()
