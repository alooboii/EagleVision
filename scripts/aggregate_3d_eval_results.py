from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ID_BY_TASK = {
    "single_frame_depth": "frame_id",
    "cycle": "pair_id",
    "reprojection": "pair_id",
    "view_warping": "pair_id",
    "normals": "frame_id",
    "pointcloud": "scene_id",
}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate arbitrary 3D-from-X task CSVs.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    return parser


def _read(path: Path) -> list[dict[str, Any]]:
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


def _metric_columns(rows: list[dict[str, Any]]) -> list[str]:
    skip = {"method", "frame_id", "pair_id", "scene_id", "difficulty", "status"}
    metrics = []
    for key in rows[0]:
        if key in skip:
            continue
        if any(math.isfinite(_float(row.get(key))) for row in rows):
            metrics.append(key)
    return metrics


def _bootstrap(values: list[float], rng: random.Random, iters: int) -> tuple[float, float]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], vals[0]
    samples = []
    for _ in range(iters):
        draw = [vals[rng.randrange(len(vals))] for _ in vals]
        samples.append(sum(draw) / len(draw))
    samples.sort()
    return samples[int(0.025 * (len(samples) - 1))], samples[int(0.975 * (len(samples) - 1))]


def _write(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = build_argparser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for path in args.inputs for row in _read(path)]
    if not rows:
        raise ValueError("No rows to aggregate")
    metrics = _metric_columns(rows)
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row.get("method", "unknown")].append(row)
    if args.baseline not in by_method:
        raise ValueError(f"Baseline '{args.baseline}' not found; available={sorted(by_method)}")
    main_rows = []
    rng = random.Random(args.seed)
    for method, method_rows in sorted(by_method.items()):
        out = {"method": method, "num_rows": len(method_rows)}
        for metric in metrics:
            if any(row.get("scene_id") for row in method_rows):
                scene_values = []
                by_scene: dict[str, list[float]] = defaultdict(list)
                for row in method_rows:
                    by_scene[row.get("scene_id", "")].append(_float(row.get(metric)))
                scene_values = [_mean(values) for values in by_scene.values()]
                out[metric] = _mean(scene_values)
                lo, hi = _bootstrap(scene_values, rng, args.bootstrap_iters)
                out[f"{metric}_ci_low"] = lo
                out[f"{metric}_ci_high"] = hi
            else:
                out[metric] = _mean([_float(row.get(metric)) for row in method_rows])
        main_rows.append(out)
    id_key = ID_BY_TASK.get(args.task, "pair_id")
    base_by_id = {row.get(id_key): row for row in by_method[args.baseline] if row.get(id_key)}
    paired = []
    for method, method_rows in by_method.items():
        if method == args.baseline:
            continue
        for row in method_rows:
            key = row.get(id_key)
            base = base_by_id.get(key)
            if base is None:
                continue
            out = {"method": method, "baseline": args.baseline, id_key: key, "scene_id": row.get("scene_id", "")}
            for metric in metrics:
                v = _float(row.get(metric))
                b = _float(base.get(metric))
                out[f"{metric}_delta"] = v - b if math.isfinite(v) and math.isfinite(b) else float("nan")
            paired.append(out)
    bucket_rows = []
    if any(row.get("difficulty") for row in rows):
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[(row.get("method", "unknown"), row.get("difficulty", ""))].append(row)
        for (method, bucket), group in groups.items():
            out = {"method": method, "difficulty": bucket, "num_rows": len(group)}
            for metric in metrics:
                out[metric] = _mean([_float(row.get(metric)) for row in group])
            bucket_rows.append(out)
    main_fields = ["method", "num_rows"] + [field for metric in metrics for field in (metric, f"{metric}_ci_low", f"{metric}_ci_high")]
    _write(args.out_dir / "main_table.csv", main_rows, main_fields)
    _write(args.out_dir / "paired_deltas.csv", paired, ["method", "baseline", id_key, "scene_id"] + [f"{metric}_delta" for metric in metrics])
    if bucket_rows:
        _write(args.out_dir / "bucket_summary.csv", bucket_rows, ["method", "difficulty", "num_rows"] + metrics)
    (args.out_dir / "main_table.md").write_text("| " + " | ".join(main_fields) + " |\n| " + " | ".join(["---"] * len(main_fields)) + " |\n" + "\n".join("| " + " | ".join(str(row.get(field, "")) for field in main_fields) + " |" for row in main_rows) + "\n", encoding="utf-8")
    (args.out_dir / "summary.md").write_text(f"# 3D Eval Aggregation\n\n- Task: `{args.task}`\n- Baseline: `{args.baseline}`\n- Inputs: {len(args.inputs)}\n- Metrics: {', '.join(metrics)}\n", encoding="utf-8")
    print(f"Wrote aggregation to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
