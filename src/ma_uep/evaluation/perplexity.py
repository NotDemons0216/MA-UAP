"""NLL / PPL 评估。"""

from __future__ import annotations

import math

import torch

from ..model.dataset import EvalSet


def evaluate(model, eval_set: EvalSet, max_windows: int | None = None) -> dict:
    windows = eval_set.windows[:max_windows] if max_windows else eval_set.windows
    labels = eval_set.labels[:max_windows] if max_windows else eval_set.labels
    loss_acc = torch.zeros((), dtype=torch.float64, device="cuda:0")
    tokens = 0
    nan_batches = 0
    for index, (begin, end, _) in enumerate(windows):
        with torch.inference_mode():
            out = model(input_ids=eval_set.input_ids[:, begin:end],
                        labels=labels[index], use_cache=False)
        count = int((labels[index][:, 1:] != -100).sum().item())
        loss = out.loss.double()
        if not torch.isfinite(loss):
            nan_batches += 1
        loss_acc += loss * count
        tokens += count
        del out
    nll = float(loss_acc.item()) / max(tokens, 1)
    finite = math.isfinite(nll)
    return {
        "nll": nll,
        "ppl": math.exp(nll) if finite and nll < 700 else float("inf"),
        "tokens": tokens,
        "windows": len(windows),
        "nan_batches": nan_batches,
        "finite": finite,
        "eval_source": eval_set.source,
    }


def delta_nll(nll: float, baseline: float) -> float:
    return nll - baseline
