"""真数据验证：不加载模型，直接从 safetensors 读权重，验证 Smax 扫描与等级分类。

用途：在显存/内存不足、无法加载完整模型时，仍能验证 importance 链路的正确性。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ma_uep.importance.scan import scan_tensor  # noqa: E402

MODEL = ROOT / "models" / "Qwen2.5-1.5B-Base" / "model.safetensors"
V4 = ROOT / "llm_weight_fault_lab" / "mauep_v0_4" / "outputs" / "qwen25_15b" / "gram"
TASK1 = ROOT / "task1_weight_importance" / "outputs" / "weight_importance"
OLD_ALLOC = ROOT / "ECC" / "outputs" / "allocation"
THRESHOLDS = ROOT / "ECC" / "outputs" / "thresholds" / "tensor_thresholds.json"

CASES = [(0, "k_proj"), (2, "k_proj"), (15, "k_proj")]


class SafetensorsManager:
    """按需从 safetensors 取张量，避免加载整个模型。"""

    def __init__(self, path: Path):
        from safetensors import safe_open
        self.path = path
        self._handle = None

    def read_raw(self, spec):
        import torch
        from safetensors import safe_open
        key = f"model.layers.{spec.block_id}.self_attn.{spec.module}.weight" \
            if spec.module in ("q_proj", "k_proj", "v_proj", "o_proj") \
            else f"model.layers.{spec.block_id}.mlp.{spec.module}.weight"
        with safe_open(str(self.path), framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(key)
        return tensor.view(torch.int16).reshape(-1).numpy().view(np.uint16)


class Spec:
    def __init__(self, block_id, module, n_weights, shape):
        self.block_id = block_id
        self.module = module
        self.n_weights = n_weights
        self.shape = shape

    @property
    def name(self):
        return f"L{self.block_id}.{self.module}"



def _require_dev_artifacts(*paths):
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        print("[skip] 该脚本需要开发期的历史产物，发布版不包含：")
        for m in missing:
            print("   -", m)
        print("这些产物可由 python run.py prepare 重新生成，但目录布局与开发期不同。")
        raise SystemExit(0)

def main() -> int:
    _require_dev_artifacts(ROOT / "ECC" / "outputs" / "allocation", ROOT / "task1_weight_importance")
    _require_dev_artifacts(ROOT / "ECC" / "outputs" / "allocation")
    manager = SafetensorsManager(MODEL)
    thresholds = json.loads(THRESHOLDS.read_text(encoding="utf-8"))["thresholds"]
    alloc_cfg = json.loads((ROOT / "ECC" / "outputs" / "allocation" /
                            "allocation_summary.json").read_text(encoding="utf-8")) \
        if (ROOT / "ECC" / "outputs" / "allocation" /
            "allocation_summary.json").exists() else {}
    out_dir = ROOT / "outputs" / "verify_scan"
    out_dir.mkdir(parents=True, exist_ok=True)

    from ma_uep.constants import BCH_HIGH
    from ma_uep.importance.ranker import GlobalRanker
    from ma_uep.importance.source import ImportanceSource

    # 用现有全局阈值（来自历史产物）判定等级，便于逐权重比对
    hi = 9.404956e-01
    med = 4.150179e-01

    report = []
    for block_id, module in CASES:
        gram = np.load(V4 / f"L{block_id}_{module}_gram.npz")
        F = gram["F"].astype(np.float64)
        K = int(F.shape[0])
        with __import__("safetensors").safe_open(str(MODEL), framework="pt",
                                                 device="cpu") as handle:
            key = (f"model.layers.{block_id}.self_attn.{module}.weight"
                   if module in ("q_proj", "k_proj", "v_proj", "o_proj")
                   else f"model.layers.{block_id}.mlp.{module}.weight")
            weight = handle.get_tensor(key)
        n = int(weight.numel())
        spec = Spec(block_id, module, n, tuple(weight.shape))

        target = out_dir / f"L{block_id}_{module}"
        target.mkdir(parents=True, exist_ok=True)
        meta = scan_tensor(manager, spec, F, target, chunk=50_000)
        smax_new = np.load(target / "I_max_log10.npy")

        old_log = np.load(TASK1 / f"L{block_id}_{module}" / "I_max_log10.npy")
        diff = np.abs(smax_new.astype(np.float64) - old_log.astype(np.float64))
        finite = np.isfinite(diff)
        max_diff = float(diff[finite].max()) if finite.any() else float("inf")

        old_level = np.load(OLD_ALLOC / f"L{block_id}_{module}" / "level_map.npy")
        with np.errstate(divide="ignore"):
            smax_val = np.power(10.0, smax_new.astype(np.float64))
        new_level = np.zeros(n, dtype=np.uint8)
        new_level[smax_val >= med] = 1
        new_level[smax_val >= hi] = 2
        agree = float((new_level == old_level).mean())

        report.append({
            "tensor": spec.name, "n_weights": n,
            "max_abs_diff_log10": max_diff,
            "level_agreement": agree,
            "counts_old": np.bincount(old_level, minlength=3).tolist(),
            "counts_new": np.bincount(new_level, minlength=3).tolist(),
            "scan_seconds": meta["seconds"],
        })
        del weight, smax_new, old_log, old_level, new_level
        print(f"{spec.name}: log10 最大差 {max_diff:.3e}，"
              f"分类一致率 {agree*100:.4f}%", flush=True)

    out = ROOT / "outputs" / "verify_scan" / "scan_verification.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 汇总 ===")
    for row in report:
        print(f"{row['tensor']:16s} log10差异 {row['max_abs_diff_log10']:.2e}  "
              f"一致率 {row['level_agreement']*100:.4f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
