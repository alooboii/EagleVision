from __future__ import annotations

import csv
import subprocess
import sys


def _write_rows(path, method: str, offset: float) -> None:
    fields = [
        "method",
        "sample_id",
        "sequence_id",
        "frame_id",
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(2):
            writer.writerow(
                {
                    "method": method,
                    "sample_id": f"s{index}",
                    "sequence_id": f"seq{index}",
                    "frame_id": str(index),
                    "absrel": 1.0 + offset + index,
                    "sqrel": 1.0,
                    "rmse": 1.0,
                    "rmse_log": 1.0,
                    "silog": 1.0,
                    "delta1": 0.5,
                    "delta2": 0.5,
                    "delta3": 0.5,
                    "photo_l1": 0.1,
                    "photo_ssim_l1": 0.1,
                    "cycle_rgb_l1": 0.1,
                    "valid_reproj_ratio": 1.0,
                    "inference_ms": 2.0,
                    "peak_gpu_mem_mb": "nan",
                }
            )


def test_aggregate_endoscopy_results_outputs_files(tmp_path):
    baseline = tmp_path / "baseline.csv"
    adapted = tmp_path / "adapted.csv"
    out_dir = tmp_path / "agg"
    _write_rows(baseline, "frozen_da2s", 0.0)
    _write_rows(adapted, "eaglevision_da2s_cycle", -0.5)

    subprocess.run(
        [
            sys.executable,
            "scripts/aggregate_endoscopy_results.py",
            "--inputs",
            str(baseline),
            str(adapted),
            "--baseline",
            "frozen_da2s",
            "--out-dir",
            str(out_dir),
            "--bootstrap-iters",
            "10",
        ],
        check=True,
    )

    assert (out_dir / "main_table.csv").exists()
    assert (out_dir / "main_table.md").exists()
    assert (out_dir / "paired_deltas.csv").exists()
    assert (out_dir / "bootstrap_ci.csv").exists()
    assert (out_dir / "summary.md").exists()
