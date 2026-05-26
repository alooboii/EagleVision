from __future__ import annotations

import argparse
import csv
import math
import random
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
    "peak_gpu_mem_mb",
]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate endoscopy evaluation CSVs.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    return parser


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _mean(values: list[float]) -> float:
    valid = [value for value in values if math.isfinite(value)]
    return sum(valid) / len(valid) if valid else float("nan")


def _sequence_means(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("sequence_id", ""))].append(_float(row.get(metric)))
    return {sequence: _mean(values) for sequence, values in grouped.items()}


def _bootstrap_ci(values: list[float], rng: random.Random, iters: int) -> tuple[float, float]:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], clean[0]
    samples = []
    for _ in range(iters):
        draw = [clean[rng.randrange(len(clean))] for _ in clean]
        samples.append(sum(draw) / len(draw))
    samples.sort()
    low_index = int(0.025 * (len(samples) - 1))
    high_index = int(0.975 * (len(samples) - 1))
    return samples[low_index], samples[high_index]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_table(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            cells.append(f"{value:.6g}" if isinstance(value, float) and math.isfinite(value) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_argparser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for path in args.inputs for row in _read_csv(path)]
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)

    rng = random.Random(args.seed)
    main_rows: list[dict[str, Any]] = []
    ci_rows: list[dict[str, Any]] = []
    for method, method_rows in sorted(by_method.items()):
        main_row: dict[str, Any] = {"method": method, "num_samples": len(method_rows)}
        for metric in METRICS:
            sample_mean = _mean([_float(row.get(metric)) for row in method_rows])
            seq_values = list(_sequence_means(method_rows, metric).values())
            seq_mean = _mean(seq_values)
            ci_low, ci_high = _bootstrap_ci(seq_values, rng, args.bootstrap_iters)
            main_row[f"{metric}_sample_mean"] = sample_mean
            main_row[f"{metric}_sequence_mean"] = seq_mean
            ci_rows.append({"method": method, "metric": metric, "ci_low": ci_low, "ci_high": ci_high})
        main_rows.append(main_row)

    baseline_rows = {row["sample_id"]: row for row in by_method.get(args.baseline, [])}
    paired_rows: list[dict[str, Any]] = []
    for method, method_rows in sorted(by_method.items()):
        if method == args.baseline:
            continue
        for row in method_rows:
            base = baseline_rows.get(row["sample_id"])
            if base is None:
                continue
            paired: dict[str, Any] = {"method": method, "baseline": args.baseline, "sample_id": row["sample_id"], "sequence_id": row.get("sequence_id", "")}
            for metric in METRICS:
                value = _float(row.get(metric))
                base_value = _float(base.get(metric))
                paired[f"{metric}_delta"] = value - base_value if math.isfinite(value) and math.isfinite(base_value) else float("nan")
            paired_rows.append(paired)

    main_fields = ["method", "num_samples"] + [f"{metric}_{kind}" for metric in METRICS for kind in ("sample_mean", "sequence_mean")]
    paired_fields = ["method", "baseline", "sample_id", "sequence_id"] + [f"{metric}_delta" for metric in METRICS]
    _write_csv(args.out_dir / "main_table.csv", main_rows, main_fields)
    _write_csv(args.out_dir / "paired_deltas.csv", paired_rows, paired_fields)
    _write_csv(args.out_dir / "bootstrap_ci.csv", ci_rows, ["method", "metric", "ci_low", "ci_high"])
    _write_markdown_table(args.out_dir / "main_table.md", main_rows, main_fields)

    summary = [
        "# Endoscopy Head-to-Head Summary",
        "",
        f"- Inputs: {len(args.inputs)} CSV files",
        f"- Methods: {', '.join(sorted(by_method))}",
        f"- Baseline: `{args.baseline}`",
        f"- Paired comparisons: {len(paired_rows)} samples",
        "",
        "See `main_table.csv`, `paired_deltas.csv`, and `bootstrap_ci.csv`.",
    ]
    (args.out_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Wrote aggregated results to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
