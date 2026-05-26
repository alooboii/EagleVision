from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

from eaglevision.data.endoscopy_collate import endoscopy_stereo_collate
from eaglevision.data.endoscopy_stereo_dataset import EndoscopyStereoDataset


def test_endoscopy_stereo_dataset_loads_tiny_manifest(tmp_path):
    left = np.zeros((4, 6, 3), dtype=np.uint8)
    right = np.ones((4, 6, 3), dtype=np.uint8) * 127
    Image.fromarray(left).save(tmp_path / "left.png")
    Image.fromarray(right).save(tmp_path / "right.png")
    Image.fromarray(np.ones((4, 6), dtype=np.uint8) * 255).save(tmp_path / "mask.png")
    np.save(tmp_path / "depth.npy", np.ones((4, 6), dtype=np.float32) * 2.0)
    np.save(tmp_path / "disp.npy", np.ones((4, 6), dtype=np.float32) * 4.0)
    np.savetxt(tmp_path / "K_left.txt", np.array([[6.0, 0.0, 3.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]], dtype=np.float32))
    np.savetxt(tmp_path / "K_right.txt", np.eye(3, dtype=np.float32))
    np.savetxt(tmp_path / "T.txt", np.eye(4, dtype=np.float32))
    row = {
        "sample_id": "seq_frame",
        "sequence_id": "seq",
        "frame_id": "frame",
        "left_rgb": "left.png",
        "right_rgb": "right.png",
        "depth": "depth.npy",
        "disparity": "disp.npy",
        "mask": "mask.png",
        "K_left": "K_left.txt",
        "K_right": "K_right.txt",
        "T_left_to_right": "T.txt",
        "baseline": 0.005,
        "focal_length": 700.0,
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

    dataset = EndoscopyStereoDataset(manifest, root=tmp_path, image_size=(2, 3))
    sample = dataset[0]

    assert sample["left_rgb"].shape == (3, 2, 3)
    assert sample["right_rgb"].shape == (3, 2, 3)
    assert sample["depth_gt"].shape == (2, 3)
    assert sample["disparity_gt"].shape == (2, 3)
    assert torch.allclose(sample["disparity_gt"], torch.full((2, 3), 2.0))
    assert sample["valid_mask"].dtype == torch.bool
    assert torch.allclose(sample["K_left"], torch.tensor([[3.0, 0.0, 1.5], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]))
    assert sample["sample_id"] == "seq_frame"

    batch = endoscopy_stereo_collate([sample])
    assert batch["left_rgb"].shape == (1, 3, 2, 3)
    assert batch["sample_id"] == ["seq_frame"]


def test_endoscopy_collate_rejects_mixed_optional_fields():
    samples = [
        {
            "sample_id": "with_depth",
            "left_rgb": torch.zeros(3, 2, 2),
            "right_rgb": torch.zeros(3, 2, 2),
            "depth_gt": torch.ones(2, 2),
        },
        {
            "sample_id": "without_depth",
            "left_rgb": torch.zeros(3, 2, 2),
            "right_rgb": torch.zeros(3, 2, 2),
        },
    ]

    with pytest.raises(ValueError, match="depth_gt.*without_depth"):
        endoscopy_stereo_collate(samples)
