from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image

from eaglevision.data.rgbd_collate import rgbd_frame_collate
from eaglevision.data.rgbd_frame_dataset import RGBDFrameDataset


def test_rgbd_frame_dataset_loads_and_resizes(tmp_path):
    Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8)).save(tmp_path / "rgb.png")
    Image.fromarray((np.ones((4, 6), dtype=np.uint16) * 2000)).save(tmp_path / "depth.png")
    np.savetxt(tmp_path / "pose.txt", np.eye(4, dtype=np.float32))
    np.savetxt(tmp_path / "K.txt", np.array([[6.0, 0.0, 3.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]], dtype=np.float32))
    row = {
        "frame_id": "scene0000_00/000000",
        "scene_id": "scene0000_00",
        "rgb": "rgb.png",
        "depth": "depth.png",
        "pose": "pose.txt",
        "intrinsics": "K.txt",
    }
    manifest = tmp_path / "frames.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

    dataset = RGBDFrameDataset(manifest, root=tmp_path, image_size=(2, 3), depth_scale=1000.0)
    sample = dataset[0]

    assert sample["rgb"].shape == (3, 2, 3)
    assert sample["depth_gt"].shape == (2, 3)
    assert torch.allclose(sample["depth_gt"], torch.full((2, 3), 2.0))
    assert torch.allclose(sample["intrinsics"], torch.tensor([[3.0, 0.0, 1.5], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]))
    assert sample["pose"].shape == (4, 4)

    batch = rgbd_frame_collate([sample])
    assert batch["rgb"].shape == (1, 3, 2, 3)
    assert batch["frame_id"] == ["scene0000_00/000000"]


def test_rgbd_collate_rejects_mixed_optional_fields():
    samples = [
        {"frame_id": "a", "scene_id": "s", "rgb": torch.zeros(3, 2, 2), "depth_gt": torch.ones(2, 2)},
        {"frame_id": "b", "scene_id": "s", "rgb": torch.zeros(3, 2, 2)},
    ]

    try:
        rgbd_frame_collate(samples)
    except ValueError as exc:
        assert "depth_gt" in str(exc)
        assert "b" in str(exc)
    else:
        raise AssertionError("mixed optional fields should raise")
