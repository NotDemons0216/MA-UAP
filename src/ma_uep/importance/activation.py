"""采集各 Linear 输入通道的激活 RMS（F_k）。

F_k = sqrt( (1/N) · Σ_t x_{t,k}² )

N 是全部评估窗口的 token 实例数（与 V0.5 口径一致：stride=128、max_length=256）。
实现为在目标模块上注册 forward_pre_hook，逐窗口累加平方和（float64）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def collect_F(model_manager, eval_set, specs, verbose: bool = True) -> dict:
    """返回 {tensor_name: np.ndarray(K,)}，K 为该 Linear 的输入维度。"""
    specs = list(specs)
    accum = {}
    counts = {}
    handles = []

    def make_hook(name: str):
        def hook(module, args):
            tensor = args[0] if isinstance(args, (tuple, list)) else args
            if tensor.dim() == 3:
                flat = tensor.reshape(-1, tensor.shape[-1])
            elif tensor.dim() == 2:
                flat = tensor
            else:
                return
            block = accum[name]
            block += flat.double().pow(2).sum(dim=0).cpu().numpy()
            counts[name] += int(flat.shape[0])
        return hook

    for spec in specs:
        module = model_manager.get_module(spec.block_id, spec.module)
        K = int(module.weight.shape[1])
        accum[spec.name] = np.zeros(K, dtype=np.float64)
        counts[spec.name] = 0
        handles.append(module.register_forward_pre_hook(make_hook(spec.name)))

    try:
        for index, (begin, end, _) in enumerate(eval_set.windows):
            with torch.inference_mode():
                model_manager.model(input_ids=eval_set.input_ids[:, begin:end],
                                    use_cache=False)
            if verbose and (index + 1) % 500 == 0:
                print(f"  F 采集 {index + 1}/{len(eval_set.windows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    result = {}
    for spec in specs:
        n = max(counts[spec.name], 1)
        result[spec.name] = np.sqrt(accum[spec.name] / n)
    return result


def save_F(F_by_tensor: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {name.replace(".", "_"): value for name, value in F_by_tensor.items()}
    path = out_dir / "F.npz"
    np.savez_compressed(path, **payload)
    return path


def load_F(path: str | Path) -> dict:
    data = np.load(Path(path))
    return {key.replace("_proj", ".", 1) if False else key: data[key]
            for key in data.files}
