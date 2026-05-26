from __future__ import annotations

import csv
import subprocess
import sys


def test_run_3d_from_x_suite_dry_writes_status(tmp_path):
    out_root = tmp_path / "suite"
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"not-used-in-dry-run")

    subprocess.run(
        [
            sys.executable,
            "scripts/run_3d_from_x_suite.py",
            "--root",
            str(tmp_path / "scannet"),
            "--checkpoint",
            str(checkpoint),
            "--out-root",
            str(out_root),
            "--max-scenes",
            "1",
            "--frames-per-scene",
            "1",
            "--dry-run",
        ],
        check=True,
    )

    with (out_root / "run_status.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {row["status"] for row in rows} == {"planned"}
    assert (out_root / "run_summary.md").exists()
