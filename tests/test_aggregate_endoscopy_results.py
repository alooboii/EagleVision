from __future__ import annotations

import csv
import json
import math
import subprocess
import sys


FIELDS = [
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
]


def _write_rows(path, method: str, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for index, row in enumerate(rows):
            payload = {
                "method": method,
                "sample_id": f"s{index}",
                "sequence_id": f"seq{index % 2}",
                "frame_id": str(index),
                "absrel": 1.0,
                "sqrel": 1.0,
                "rmse": 2.0,
                "rmse_log": 0.5,
                "silog": 10.0,
                "delta1": 0.5,
                "delta2": 0.7,
                "delta3": 0.9,
                "photo_l1": 0.2,
                "photo_ssim_l1": 0.3,
                "cycle_rgb_l1": 0.4,
                "valid_reproj_ratio": 0.8,
                "inference_ms": 5.0,
            }
            payload.update(row)
            writer.writerow(payload)


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_aggregate_endoscopy_results_deltas_ranking_and_nans(tmp_path):
    frozen = tmp_path / "frozen.csv"
    eagle = tmp_path / "eagle.csv"
    lora = tmp_path / "lora.csv"
    param_summary = tmp_path / "eaglevision_da2s_cycle" / "param_summary.json"
    param_summary.parent.mkdir()
    param_summary.write_text(json.dumps({"adapter_trainable": 1234}) + "\n", encoding="utf-8")
    out_dir = tmp_path / "agg"
    _write_rows(
        frozen,
        "frozen_da2s",
        [
            {"absrel": 1.0, "rmse": 2.0, "silog": 10.0, "delta1": 0.50, "photo_l1": 0.20, "inference_ms": 5.0},
            {"absrel": 3.0, "rmse": 4.0, "silog": 30.0, "delta1": 0.70, "photo_l1": 0.40, "inference_ms": 7.0},
        ],
    )
    _write_rows(
        eagle,
        "eaglevision_da2s_cycle",
        [
            {"absrel": 0.5, "rmse": 1.0, "silog": 5.0, "delta1": 0.80, "photo_l1": 0.10, "inference_ms": 6.0},
            {"absrel": "nan", "rmse": 3.0, "silog": 15.0, "delta1": 0.90, "photo_l1": 0.20, "inference_ms": 8.0},
        ],
    )
    _write_rows(
        lora,
        "dares_style_lora_da2s_cycle",
        [
            {"absrel": 1.5, "rmse": 2.5, "silog": 11.0, "delta1": 0.60, "photo_l1": 0.30, "inference_ms": 9.0},
            {"absrel": 2.5, "rmse": 3.5, "silog": 19.0, "delta1": 0.65, "photo_l1": 0.35, "inference_ms": 11.0},
        ],
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/aggregate_endoscopy_results.py",
            "--inputs",
            str(frozen),
            str(eagle),
            str(lora),
            "--baseline",
            "frozen_da2s",
            "--out-dir",
            str(out_dir),
            "--param-summaries",
            str(param_summary),
        ],
        check=True,
    )

    summary = {row["method"]: row for row in _read_csv(out_dir / "summary.csv")}
    assert (out_dir / "summary.md").exists()
    assert (out_dir / "per_sequence_summary.csv").exists()
    assert (out_dir / "method_ranking.md").exists()

    assert math.isclose(float(summary["frozen_da2s"]["absrel"]), 2.0)
    assert math.isclose(float(summary["eaglevision_da2s_cycle"]["absrel"]), 0.5)
    assert math.isclose(float(summary["eaglevision_da2s_cycle"]["delta_vs_frozen_da2s_absrel"]), -1.5)
    assert math.isclose(float(summary["eaglevision_da2s_cycle"]["delta_vs_frozen_da2s_delta1"]), 0.25)
    assert math.isclose(float(summary["dares_style_lora_da2s_cycle"]["delta_vs_frozen_da2s_inference_ms"]), 4.0)

    ranking = (out_dir / "method_ranking.md").read_text(encoding="utf-8")
    summary_md = (out_dir / "summary.md").read_text(encoding="utf-8")
    absrel_section = ranking.split("## absrel", maxsplit=1)[1].split("## sqrel", maxsplit=1)[0]
    delta1_section = ranking.split("## delta1", maxsplit=1)[1].split("## delta2", maxsplit=1)[0]
    assert "| 1 | eaglevision_da2s_cycle | 0.5 |" in absrel_section
    assert "| 1 | eaglevision_da2s_cycle | 0.85 |" in delta1_section
    assert "| eaglevision_da2s_cycle | 1234 |" in summary_md
