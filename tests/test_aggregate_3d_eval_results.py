from __future__ import annotations

import csv
import math
import subprocess
import sys


def _write(path, method: str, values: list[float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "pair_id", "scene_id", "difficulty", "absrel", "delta1"])
        writer.writeheader()
        for idx, value in enumerate(values):
            writer.writerow({"method": method, "pair_id": f"p{idx}", "scene_id": f"s{idx % 2}", "difficulty": "easy", "absrel": value, "delta1": 1.0 - value})


def test_aggregate_3d_eval_results_deltas_and_nan_handling(tmp_path):
    frozen = tmp_path / "frozen.csv"
    adapted = tmp_path / "adapted.csv"
    _write(frozen, "frozen", [1.0, 2.0])
    _write(adapted, "adapted", [0.5, float("nan")])
    out_dir = tmp_path / "agg"

    subprocess.run([sys.executable, "scripts/aggregate_3d_eval_results.py", "--task", "reprojection", "--inputs", str(frozen), str(adapted), "--baseline", "frozen", "--out-dir", str(out_dir), "--bootstrap-iters", "10"], check=True)

    with (out_dir / "main_table.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = {row["method"]: row for row in csv.DictReader(handle)}
    with (out_dir / "paired_deltas.csv").open("r", encoding="utf-8", newline="") as handle:
        deltas = list(csv.DictReader(handle))
    assert math.isclose(float(rows["frozen"]["absrel"]), 1.5)
    assert math.isclose(float(rows["adapted"]["absrel"]), 0.5)
    assert math.isclose(float(deltas[0]["absrel_delta"]), -0.5)
    assert (out_dir / "bucket_summary.csv").exists()
    assert (out_dir / "summary.md").exists()
