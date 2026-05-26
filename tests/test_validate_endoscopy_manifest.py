from __future__ import annotations

import json

import numpy as np
from PIL import Image

from eaglevision.cli.validate_endoscopy_manifest import validate_manifest


def _write_image(path, value: int = 0) -> None:
    array = np.full((4, 6, 3), value, dtype=np.uint8)
    Image.fromarray(array).save(path)


def test_validate_endoscopy_manifest_valid_fake_images(tmp_path):
    _write_image(tmp_path / "left0.png", 0)
    _write_image(tmp_path / "right0.png", 10)
    rows = [
        {
            "sample_id": "s0",
            "sequence_id": "seq0",
            "left_rgb": "left0.png",
            "right_rgb": "right0.png",
            "baseline": 0.005,
            "focal_length": 700.0,
        }
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = validate_manifest(manifest, root=tmp_path, require_geometry=True)

    assert summary["valid"] is True
    assert summary["num_samples"] == 1
    assert summary["has_baseline_focal"] is True
    assert summary["image_size_stats"]["unique_sizes"] == ["6x4"]
    assert summary["sequences"] == {"seq0": 1}


def test_validate_endoscopy_manifest_flags_mixed_optional_fields(tmp_path):
    _write_image(tmp_path / "left0.png", 0)
    _write_image(tmp_path / "right0.png", 10)
    _write_image(tmp_path / "left1.png", 20)
    _write_image(tmp_path / "right1.png", 30)
    np.save(tmp_path / "depth.npy", np.ones((4, 6), dtype=np.float32))
    rows = [
        {
            "sample_id": "s0",
            "sequence_id": "seq0",
            "left_rgb": "left0.png",
            "right_rgb": "right0.png",
            "depth": "depth.npy",
            "baseline": 0.005,
            "focal_length": 700.0,
        },
        {
            "sample_id": "s1",
            "sequence_id": "seq0",
            "left_rgb": "left1.png",
            "right_rgb": "right1.png",
            "baseline": 0.005,
            "focal_length": 700.0,
        },
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = validate_manifest(manifest, root=tmp_path, require_geometry=True)

    assert summary["valid"] is False
    assert "depth" in summary["inconsistent_fields"]
    assert summary["inconsistent_fields"]["depth"]["missing_sample_ids"] == ["s1"]
