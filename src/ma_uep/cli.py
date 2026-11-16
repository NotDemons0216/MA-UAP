"""MA-UEP V1.0 统一命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .constants import (BLOCKS, LEVEL_RATIOS, MODEL_NAME, MODULES,
                        SYSTEM_VERSION)
from .paths import PROJECT_ROOT
from .experiment.config import ExperimentConfig
from .experiment.registry import ExperimentRegistry, new_experiment_id

REPO_ROOT = PROJECT_ROOT


def _load_config(args) -> ExperimentConfig:
    if args.config:
        cfg = ExperimentConfig.load(args.config)
    else:
        default = REPO_ROOT / "configs" / "v1_default.yaml"
        cfg = ExperimentConfig.load(default) if default.exists() \
            else ExperimentConfig()
    if getattr(args, "mode", None):
        cfg.mode = args.mode
    if getattr(args, "ber", None) is not None:
        cfg.channel.mode = "ber"
        cfg.channel.ber = args.ber
    if getattr(args, "seed", None) is not None:
        cfg.channel.seed = args.seed
    if getattr(args, "max_windows", None):
        cfg.max_windows = args.max_windows
    if getattr(args, "guard", None) is not None:
        cfg.anomaly_guard.enabled = args.guard
    cfg.validate()
    return cfg


def cmd_config(args) -> int:
    cfg = _load_config(args)
    print(cfg.snapshot())
    return 0


def cmd_check(args) -> int:
    """检查模型、重要性数据、阈值文件是否齐备。"""
    cfg = _load_config(args)
    report = {"system": SYSTEM_VERSION, "model": MODEL_NAME,
              "blocks": BLOCKS, "modules": MODULES,
              "ratios": LEVEL_RATIOS, "checks": {}}
    checks = report["checks"]

    model_path = Path(cfg.model.path)
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path
    checks["model_path"] = {"path": str(model_path), "exists": model_path.exists()}

    importance = Path(cfg.importance.root)
    if not importance.is_absolute():
        importance = REPO_ROOT / importance
    checks["importance_root"] = {"path": str(importance), "exists": importance.exists()}

    allocation = Path(cfg.importance.allocation_dir)
    if not allocation.is_absolute():
        allocation = REPO_ROOT / allocation
    checks["allocation_dir"] = {"path": str(allocation), "exists": allocation.exists()}

    thresholds = Path(cfg.anomaly_guard.thresholds)
    if not thresholds.is_absolute():
        thresholds = REPO_ROOT / thresholds
    checks["thresholds"] = {"path": str(thresholds), "exists": thresholds.exists()}

    bch = Path(cfg.ecc.bch_path) if cfg.ecc.bch_path else REPO_ROOT / "BCH"
    checks["bch_path"] = {"path": str(bch), "exists": bch.exists()}

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(item["exists"] for item in checks.values()) else 1


def cmd_run(args) -> int:
    from .pipeline.runner import ExperimentRunner
    cfg = _load_config(args)
    registry = ExperimentRegistry(REPO_ROOT / cfg.output.root, REPO_ROOT)
    params = {"ber": cfg.channel.ber if cfg.channel.mode == "ber" else None,
              "k_bits": cfg.channel.k_bits if cfg.channel.mode == "fixed" else None,
              "seed": cfg.channel.seed}
    experiment_id = args.experiment_id or new_experiment_id(cfg.mode, params)
    path = registry.create(experiment_id)
    registry.write_meta(path, cfg, {"mode": cfg.mode, **params})

    runner = ExperimentRunner(cfg, REPO_ROOT)
    results = runner.run()
    results["mode"] = cfg.mode
    results["ber"] = params["ber"]
    results["seed"] = params["seed"]
    results["experiment_id"] = experiment_id
    registry.write_result(path, results)

    from .experiment.reporter import render_markdown, summarise, write_csv
    rows = summarise([results])
    write_csv(rows, path / "summary.csv")
    (path / "REPORT.md").write_text(render_markdown(rows, f"实验 {experiment_id}"),
                                    encoding="utf-8")

    evaluation = results.get("evaluation", {})
    print(f"[{experiment_id}] NLL={evaluation.get('nll')} "
          f"ΔNLL={evaluation.get('delta_nll')}")
    print(f"结果目录: {path}")
    return 0


def cmd_prepare(args) -> int:
    """一条龙：模型 + 数据集 → F → Smax → 等级图 + 阈值。"""
    from .importance.prepare import prepare
    cfg = _load_config(args)
    payload = prepare(cfg, REPO_ROOT, all_blocks=args.all_blocks,
                      save_risk_map=args.save_risk_map,
                      keep_smax=args.keep_smax,
                      work_dir=args.output_dir)
    print(json.dumps({
        "n_tensors": payload["n_tensors"],
        "total_weights": payload["total_weights"],
        "level_counts": payload["level_counts"],
        "timing_seconds": payload["timing_seconds"],
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_batch(args) -> int:
    import yaml
    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base = ExperimentConfig.from_dict(payload.get("base", {}))
    grid = payload.get("grid", {})
    modes = grid.get("mode") or [base.mode]
    bers = grid.get("ber") or [base.channel.ber]
    seeds = grid.get("seed") or [base.channel.seed]
    guards = grid.get("anomaly_guard")
    registry = ExperimentRegistry(REPO_ROOT / base.output.root, REPO_ROOT)
    from .experiment.reporter import render_markdown, summarise, write_csv
    from .pipeline.runner import ExperimentRunner

    all_rows = []
    for mode in modes:
        for ber in bers:
            for seed in seeds:
                for guard in (guards if guards is not None else [None]):
                    cfg = ExperimentConfig.from_dict(payload.get("base", {}))
                    cfg.mode = mode
                    cfg.channel.ber = float(ber)
                    cfg.channel.seed = int(seed)
                    if guard is not None:
                        cfg.anomaly_guard.enabled = bool(guard)
                    cfg.validate()
                    experiment_id = new_experiment_id(
                        mode, {"ber": ber, "seed": seed})
                    path = registry.create(experiment_id)
                    registry.write_meta(path, cfg, {"mode": mode, "ber": ber,
                                                    "seed": seed})
                    results = ExperimentRunner(cfg, REPO_ROOT).run()
                    results.update({"mode": mode, "ber": float(ber),
                                    "seed": int(seed),
                                    "experiment_id": experiment_id})
                    registry.write_result(path, results)
                    all_rows.extend(summarise([results]))
                    print(f"[{experiment_id}] ΔNLL={results['evaluation']['delta_nll']}")
    write_csv(all_rows, REPO_ROOT / base.output.root / "reports" / "batch_summary.csv")
    (REPO_ROOT / base.output.root / "reports" / "BATCH_REPORT.md").write_text(
        render_markdown(all_rows, "MA-UEP 批量实验"), encoding="utf-8")
    return 0


def cmd_report(args) -> int:
    from .experiment.reporter import render_markdown, summarise
    registry = ExperimentRegistry(REPO_ROOT / "MA-UEP" / "outputs", REPO_ROOT)
    results = registry.load_results()
    if args.experiment_id:
        results = [r for r in results if r.get("experiment_id") == args.experiment_id]
    rows = summarise(results)
    print(render_markdown(rows))
    return 0


def cmd_test(args) -> int:
    import subprocess
    return subprocess.call([sys.executable, "-m", "unittest", "discover",
                            "-s", str(REPO_ROOT / "tests"), "-t", str(REPO_ROOT)])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ma-uep", description=f"{SYSTEM_VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--config", help="YAML 配置路径")
        p.add_argument("--mode", help="实验模式")
        p.add_argument("--ber", type=float, help="BER")
        p.add_argument("--seed", type=int, help="随机种子")
        p.add_argument("--max-windows", type=int, default=None,
                       help="只评估前 N 个窗口（调试用）")
        p.add_argument("--guard", type=lambda x: x.lower() == "true",
                       default=None, help="是否启用异常遏制 (true/false)")

    p_config = sub.add_parser("config", help="显示配置")
    add_common(p_config)
    p_config.add_argument("--show", action="store_true")
    p_config.set_defaults(func=cmd_config)

    p_check = sub.add_parser("check", help="检查依赖与数据")
    add_common(p_check)
    p_check.set_defaults(func=cmd_check)

    p_run = sub.add_parser("run", help="执行单次实验")
    add_common(p_run)
    p_run.add_argument("--experiment-id", default=None)
    p_run.set_defaults(func=cmd_run)

    p_prepare = sub.add_parser("prepare", help="一条龙准备等级图与阈值")
    add_common(p_prepare)
    p_prepare.add_argument("--all-blocks", action="store_true",
                           help="处理全部 28 个 Block（默认只处理配置中的 Block）")
    p_prepare.add_argument("--save-risk-map", action="store_true",
                           help="额外保存 Bit-level 风险地图（约 16 GB）")
    p_prepare.add_argument("--keep-smax", action="store_true",
                           help="保留每权重 Smax 数组（约 936 MB）")
    p_prepare.add_argument("--output-dir", default=None,
                           help="产物目录，默认为 outputs/prepared")
    p_prepare.set_defaults(func=cmd_prepare)

    p_batch = sub.add_parser("batch", help="执行批量实验")
    p_batch.add_argument("--config", required=True)
    p_batch.set_defaults(func=cmd_batch)

    p_report = sub.add_parser("report", help="查看实验结果")
    p_report.add_argument("--experiment-id", default=None)
    p_report.set_defaults(func=cmd_report)

    p_test = sub.add_parser("test", help="运行单元测试")
    p_test.set_defaults(func=cmd_test)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
