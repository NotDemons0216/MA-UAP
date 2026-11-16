"""结果汇总与报告生成。"""

from __future__ import annotations

import csv
from pathlib import Path


def summarise(results: list[dict]) -> list[dict]:
    rows = []
    for item in results:
        evaluation = item.get("evaluation") or {}
        baseline = item.get("baseline") or {}
        channel = (item.get("channel") or {}).get("total", {})
        rows.append({
            "experiment_id": item.get("experiment_id", ""),
            "mode": item.get("mode", ""),
            "ber": item.get("ber", ""),
            "seed": item.get("seed", ""),
            "baseline_nll": baseline.get("nll"),
            "nll": evaluation.get("nll"),
            "delta_nll": evaluation.get("delta_nll"),
            "ppl": evaluation.get("ppl"),
            "n_errors": channel.get("n_errors"),
            "n_changed": channel.get("n_changed"),
            "code_rate": (item.get("uep") or {}).get("code_rate"),
        })
    return rows


def write_csv(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def render_markdown(rows: list[dict], title: str = "MA-UEP 实验结果") -> str:
    lines = [f"# {title}", ""]
    if not rows:
        return "\n".join(lines + ["（无结果）", ""])
    header = ["模式", "BER", "seed", "Baseline NLL", "NLL", "ΔNLL", "PPL",
              "注入错误", "改动权重"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for row in rows:
        lines.append("| " + " | ".join([
            str(row.get("mode", "")),
            f"{row['ber']:g}" if isinstance(row.get("ber"), float) else str(row.get("ber")),
            str(row.get("seed", "")),
            _fmt(row.get("baseline_nll")),
            _fmt(row.get("nll")),
            _fmt(row.get("delta_nll"), signed=True),
            _fmt(row.get("ppl")),
            str(row.get("n_errors", "")),
            str(row.get("n_changed", "")),
        ]) + " |")
    return "\n".join(lines) + "\n"


def _fmt(value, signed: bool = False) -> str:
    if value is None or value == "":
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:          # NaN
        return "NaN"
    return f"{number:+.6f}" if signed else f"{number:.6f}"
