from __future__ import annotations

import csv
import json
import subprocess
import sys

import numpy as np


def _write_pair_fixture(tmp_path):
    depths = tmp_path / "depths"
    depths.mkdir()
    np.save(depths / "a.npy", np.ones((3, 4), dtype=np.float32))
    np.save(depths / "b.npy", np.ones((3, 4), dtype=np.float32))
    np.savetxt(tmp_path / "pose_a.txt", np.eye(4, dtype=np.float32))
    np.savetxt(tmp_path / "pose_b.txt", np.eye(4, dtype=np.float32))
    np.savetxt(tmp_path / "K.txt", np.eye(3, dtype=np.float32))
    pred_manifest = tmp_path / "pred.jsonl"
    rows = [
        {"method": "frozen", "frame_id": "scene/a", "scene_id": "scene", "pred_depth": "depths/a.npy", "pose": str(tmp_path / "pose_a.txt"), "intrinsics": str(tmp_path / "K.txt")},
        {"method": "frozen", "frame_id": "scene/b", "scene_id": "scene", "pred_depth": "depths/b.npy", "pose": str(tmp_path / "pose_b.txt"), "intrinsics": str(tmp_path / "K.txt")},
    ]
    pred_manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    pair_manifest = tmp_path / "pairs.jsonl"
    pair_manifest.write_text(json.dumps({"pair_id": "p0", "scene_id": "scene", "source_frame_id": "scene/a", "target_frame_id": "scene/b", "difficulty": "easy", "translation_m": 0.0, "rotation_deg": 0.0}) + "\n", encoding="utf-8")
    return pred_manifest, pair_manifest


def test_eval_cycle_consistency_writes_requested_schema(tmp_path):
    pred_manifest, pair_manifest = _write_pair_fixture(tmp_path)
    out = tmp_path / "cycle.csv"

    subprocess.run([sys.executable, "-m", "eaglevision.cli.eval_cycle_consistency", "--pair-manifest", str(pair_manifest), "--pred-manifest", str(pred_manifest), "--out", str(out)], check=True)

    with out.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["method"] == "frozen"
    assert rows[0]["pair_id"] == "p0"
    assert "cycle_depth_l1_pred" in rows[0]
    assert rows[0]["status"] == "ok"
