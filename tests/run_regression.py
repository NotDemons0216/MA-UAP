"""Phase 4：历史回归。

对每个历史配置运行新系统，与冻结的历史 ΔNLL 对比。
历史基准（seed=42，完整 2335 窗口）：
    BER 1e-5:  none +0.010470 / ma_uep +0.003489 / ma_uep+guard +0.000975
    BER 1e-4:  none +3.360865 / ma_uep +0.032288 / ma_uep+guard +0.003874
    fixed k=3:              ma_uep +9.491330 / ma_uep+guard +0.006037
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
HISTORY = {
    ("no_protection", "ber", 1e-5, 0): 0.010470,
    ("ma_uep_without_guard", "ber", 1e-5, 0): 0.003489,
    ("ma_uep", "ber", 1e-5, 0): 0.000975,
    ("no_protection", "ber", 1e-4, 0): 3.360865,
    ("ma_uep_without_guard", "ber", 1e-4, 0): 0.032288,
    ("ma_uep", "ber", 1e-4, 0): 0.003874,
    ("ma_uep_without_guard", "fixed", 1e-4, 3): 9.491330,
    ("ma_uep", "fixed", 1e-4, 3): 0.006037,
}


def build_config(mode: str, channel_mode: str, ber: float, k_bits: int) -> dict:
    guard = mode == "ma_uep"
    return {
        "mode": mode,
        "model": {"path": "models/Qwen2.5-1.5B-Base", "device": "cuda:0"},
        "dataset": {"split": "test", "stride": 128, "max_length": 256},
        "scope": {"blocks": [0, 2, 15, 23, 27],
                  "modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"]},
        "importance": {"root": "task1_weight_importance/outputs/weight_importance",
                       "allocation_dir": "ECC/outputs/allocation"},
        "uep": {"ratios": {"high": 0.05, "medium": 0.15, "low": 0.80}},
        "anomaly_guard": {"enabled": guard,
                          "thresholds": "ECC/outputs/thresholds/tensor_thresholds.json"},
        "channel": {"mode": channel_mode, "ber": ber,
                    "bad_weights": 150000, "k_bits": k_bits, "seed": 42},
        "output": {"root": "outputs/regression"},
    }



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
    import argparse
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None,
                        help="只跑某一类：ber | fixed")
    cli = parser.parse_args()
    tmp = ROOT / "outputs" / "regression" / "_configs"
    tmp.mkdir(parents=True, exist_ok=True)
    rows = []
    for (mode, channel_mode, ber, k_bits), expected in HISTORY.items():
        if cli.only and channel_mode != cli.only:
            continue
        tag = f"{mode}_{channel_mode}_{ber:g}_k{k_bits}"
        cfg_path = tmp / f"{tag}.yaml"
        cfg_path.write_text(yaml.safe_dump(
            build_config(mode, channel_mode, ber, k_bits),
            allow_unicode=True), encoding="utf-8")
        print(f"=== {tag} ===", flush=True)
        proc = subprocess.run(
            [sys.executable, str(ROOT / "run.py"), "run", "--config", str(cfg_path)],
            cwd=str(ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        if proc.returncode != 0:
            print(proc.stdout[-800:])
            print(proc.stderr[-800:])
            rows.append({"case": tag, "expected": expected, "actual": None,
                         "diff": None, "status": "FAILED"})
            continue
        result_file = None
        for line in proc.stdout.splitlines():
            if line.startswith("结果目录:"):
                result_file = Path(line.split(":", 1)[1].strip()) / "result.json"
        if not result_file or not result_file.exists():
            rows.append({"case": tag, "expected": expected, "actual": None,
                         "diff": None, "status": "NO_RESULT"})
            continue
        payload = json.loads(result_file.read_text(encoding="utf-8"))
        actual = payload["evaluation"]["delta_nll"]
        rows.append({"case": tag, "expected": expected, "actual": actual,
                     "diff": actual - expected,
                     "status": "OK" if abs(actual - expected) < 1e-3 else "DIFF"})
        print(f"  expected={expected:+.6f} actual={actual:+.6f} "
              f"diff={actual-expected:+.2e}", flush=True)

    out = ROOT / "outputs" / "regression"
    out = ROOT / "outputs" / "regression"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"regression_{cli.only}.json" if cli.only else "regression.json"
    (out / tag).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 汇总 ===")
    for row in rows:
        print(f"{row['case']:45s} {row['status']:8s} "
              f"exp={row['expected']:+.6f} "
              f"act={row['actual'] if row['actual'] is None else format(row['actual'], '+.6f')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
