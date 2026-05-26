from __future__ import annotations

import csv
import json
import subprocess
import sys

import numpy as np
from PIL import Image


def test_eval_view_warping_writes_requested_schema(tmp_path):
    Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(tmp_path / "a.png")
    Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(tmp_path / "b.png")
    depths = tmp_path / "depths"
    depths.mkdir()
    np.save(depths / "a.npy", np.ones((3, 4), dtype=np.float32))
    np.save(depths / "b.npy", np.ones((3, 4), dtype=np.float32))
    np.savetxt(tmp_path / "pose.txt", np.eye(4, dtype=np.float32))
    np.savetxt(tmp_path / "K.txt", np.eye(3, dtype=np.float32))
    pred_manifest = tmp_path / "pred.jsonl"
    rows = [
        {"method": "adapted", "frame_id": "scene/a", "scene_id": "scene", "rgb": str(tmp_path / "a.png"), "pred_depth": "depths/a.npy", "pose": str(tmp_path / "pose.txt"), "intrinsics": str(tmp_path / "K.txt")},
        {"method": "adapted", "frame_id": "scene/b", "scene_id": "scene", "rgb": str(tmp_path / "b.png"), "pred_depth": "depths/b.npy", "pose": str(tmp_path / "pose.txt"), "intrinsics": str(tmp_path / "K.txt")},
    ]
    pred_manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    pair_manifest = tmp_path / "pairs.jsonl"
    pair_manifest.write_text(json.dumps({"pair_id": "p0", "scene_id": "scene", "source_frame_id": "scene/a", "target_frame_id": "scene/b", "difficulty": "easy", "translation_m": 0.0, "rotation_deg": 0.0}) + "\n", encoding="utf-8")
    out = tmp_path / "view.csv"

    subprocess.run([sys.executable, "-m", "eaglevision.cli.eval_view_warping", "--pair-manifest", str(pair_manifest), "--pred-manifest", str(pred_manifest), "--out", str(out)], check=True)

    with out.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["method"] == "adapted"
    assert "view_rgb_l1" in rows[0]
    assert "valid_warp_ratio" in rows[0]
    assert rows[0]["status"] == "ok"
