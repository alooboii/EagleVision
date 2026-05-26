from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


METRICS = [
    "absrel",
    "sqrel",
    "rmse",
    "rmse_log",
    "silog",
    "delta1",
    "delta2",
    "delta3",
    "photo_l1",
    "photo_ssim_l1",
    "cycle_rgb_l1",
    "valid_reproj_ratio",
    "inference_ms",
]
DELTA_METRICS = [
    "absrel",
    "rmse",
    "silog",
    "delta1",
    "photo_l1",
    "photo_ssim_l1",
    "cycle_rgb_l1",
    "valid_reproj_ratio",
    "inference_ms",
]
LOWER_IS_BETTER = {
    "absrel",
    "sqrel",
    "rmse",
    "rmse_log",
    "silog",
    "photo_l1",
    "photo_ssim_l1",
    "cycle_rgb_l1",
    "inference_ms",
}
HIGHER_IS_BETTER = {"delta1", "delta2", "delta3", "valid_reproj_ratio"}
REQUIRED_COLUMNS = {"method", *DELTA_METRICS}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate Frozen DA2 vs EagleVision vs DARES-style LoRA endoscopy CSVs.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--param-summaries", type=Path, nargs="*", default=[])
    return parser


def _float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}" if math.isfinite(value) else "NaN"
    return str(value)


def _mean(values: list[float]) -> float:
    valid = [value for value in values if math.isfinite(value)]
    return sum(valid) / len(valid) if valid else float("nan")


def read_and_validate_csvs(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            missing = sorted(REQUIRED_COLUMNS - columns)
            if missing:
                raise ValueError(f"{path} is missing required columns: {missing}")
            for row in reader:
                row["_source_csv"] = str(path)
                rows.append(row)
    if not rows:
        raise ValueError("No rows found in input CSVs")
    return rows


def aggregate_by_method(rows: list[dict[str, str]], baseline: str) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    if baseline not in by_method:
        raise ValueError(f"Baseline method '{baseline}' was not found. Available: {sorted(by_method)}")

    summary_rows: list[dict[str, Any]] = []
    baseline_means = {metric: _mean([_float(row.get(metric)) for row in by_method[baseline]]) for metric in METRICS}
    for method in sorted(by_method):
        method_rows = by_method[method]
        out: dict[str, Any] = {"method": method, "num_samples": len(method_rows)}
        for metric in METRICS:
            out[metric] = _mean([_float(row.get(metric)) for row in method_rows])
        for metric in DELTA_METRICS:
            value = out[metric]
            base_value = baseline_means[metric]
            out[f"delta_vs_{baseline}_{metric}"] = value - base_value if math.isfinite(value) and math.isfinite(base_value) else float("nan")
        summary_rows.append(out)
    return summary_rows


def aggregate_by_sequence(rows: list[dict[str, str]], baseline: str) -> list[dict[str, Any]]:
    if not any("sequence_id" in row and row.get("sequence_id") for row in rows):
        return []

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row.get("sequence_id", ""))].append(row)

    baseline_by_sequence: dict[str, dict[str, float]] = {}
    for (method, sequence_id), sequence_rows in grouped.items():
        if method == baseline:
            baseline_by_sequence[sequence_id] = {
                metric: _mean([_float(row.get(metric)) for row in sequence_rows]) for metric in METRICS
            }

    out_rows: list[dict[str, Any]] = []
    for method, sequence_id in sorted(grouped):
        sequence_rows = grouped[(method, sequence_id)]
        out: dict[str, Any] = {"method": method, "sequence_id": sequence_id, "num_samples": len(sequence_rows)}
        for metric in METRICS:
            out[metric] = _mean([_float(row.get(metric)) for row in sequence_rows])
        baseline_values = baseline_by_sequence.get(sequence_id)
        if baseline_values is not None:
            for metric in DELTA_METRICS:
                value = out[metric]
                base_value = baseline_values[metric]
                out[f"delta_vs_{baseline}_{metric}"] = value - base_value if math.isfinite(value) and math.isfinite(base_value) else float("nan")
        out_rows.append(out)
    return out_rows


def ranking_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rankings: list[dict[str, Any]] = []
    for metric in METRICS:
        reverse = metric in HIGHER_IS_BETTER
        valid = [row for row in summary_rows if math.isfinite(float(row.get(metric, float("nan"))))]
        valid.sort(key=lambda row: float(row[metric]), reverse=reverse)
        for rank, row in enumerate(valid, start=1):
            rankings.append(
                {
                    "metric": metric,
                    "rank": rank,
                    "method": row["method"],
                    "value": row[metric],
                    "direction": "higher_is_better" if reverse else "lower_is_better",
                }
            )
    return rankings


def load_param_summaries(paths: list[Path]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        trainable = (
            payload.get("trainable_param_count")
            or payload.get("adapter_trainable")
            or payload.get("trainable_lora_params")
            or payload.get("base_trainable")
        )
        summaries.append({"label": path.parent.name, "path": str(path), "trainable_param_count": trainable})
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format(row.get(column, "")) for column in columns) + " |")
    return lines


def write_summary_md(path: Path, summary_rows: list[dict[str, Any]], baseline: str, param_summaries: list[dict[str, Any]]) -> None:
    columns = ["method", "num_samples", *DELTA_METRICS, *[f"delta_vs_{baseline}_{metric}" for metric in DELTA_METRICS]]
    lines = ["# SCARED Head-to-Head Summary", "", f"Baseline: `{baseline}`", "", *_markdown_table(summary_rows, columns)]
    if param_summaries:
        lines.extend(["", "## Trainable Parameters", ""])
        lines.extend(_markdown_table(param_summaries, ["label", "trainable_param_count", "path"]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ranking_md(path: Path, rankings: list[dict[str, Any]]) -> None:
    by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rankings:
        by_metric[row["metric"]].append(row)
    lines = ["# Method Ranking", ""]
    for metric in METRICS:
        rows = by_metric.get(metric, [])
        if not rows:
            continue
        direction = rows[0]["direction"]
        lines.extend([f"## {metric}", "", f"Direction: `{direction}`", ""])
        lines.extend(_markdown_table(rows, ["rank", "method", "value"]))
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = build_argparser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_and_validate_csvs(args.inputs)
    summary_rows = aggregate_by_method(rows, args.baseline)
    sequence_rows = aggregate_by_sequence(rows, args.baseline)
    rankings = ranking_rows(summary_rows)
    param_summaries = load_param_summaries(args.param_summaries)

    summary_fields = ["method", "num_samples", *METRICS, *[f"delta_vs_{args.baseline}_{metric}" for metric in DELTA_METRICS]]
    _write_csv(args.out_dir / "summary.csv", summary_rows, summary_fields)
    write_summary_md(args.out_dir / "summary.md", summary_rows, args.baseline, param_summaries)
    if sequence_rows:
        sequence_fields = ["method", "sequence_id", "num_samples", *METRICS, *[f"delta_vs_{args.baseline}_{metric}" for metric in DELTA_METRICS]]
        _write_csv(args.out_dir / "per_sequence_summary.csv", sequence_rows, sequence_fields)
    write_ranking_md(args.out_dir / "method_ranking.md", rankings)

    print(f"Wrote aggregated comparison to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
