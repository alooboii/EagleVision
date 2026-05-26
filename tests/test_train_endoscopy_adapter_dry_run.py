from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image

from eaglevision.cli.train_endoscopy_adapter import run_dry_run


class TinyPrior(torch.nn.Module):
    def predict_depth(self, images: torch.Tensor) -> torch.Tensor:
        return torch.ones(images.shape[0], images.shape[2], images.shape[3], device=images.device)


def _write_rgb(path, shift: int = 0) -> None:
    x = np.linspace(0, 255, num=7, dtype=np.float32)[None, :, None]
    y = np.linspace(0, 64, num=5, dtype=np.float32)[:, None, None]
    image = np.repeat((x + y + shift).clip(0, 255).astype(np.uint8), 3, axis=2)
    Image.fromarray(image).save(path)


def test_train_endoscopy_adapter_dry_run_with_tiny_prior(tmp_path):
    _write_rgb(tmp_path / "left.png", 0)
    _write_rgb(tmp_path / "right.png", 20)
    row = {
        "sample_id": "s0",
        "sequence_id": "seq",
        "frame_id": "0",
        "left_rgb": "left.png",
        "right_rgb": "right.png",
        "baseline": 1.0,
        "focal_length": 2.0,
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    config = {
        "data": {"root": str(tmp_path), "image_size": [5, 7], "min_depth": 1e-3, "max_depth": 300.0},
        "base_model": {},
        "adapter": {"hidden_channels": 4, "residual_scale": 0.2, "use_tanh": False},
        "loss": {
            "weights": {
                "photo": 1.0,
                "cycle_rgb": 1.0,
                "prior_log_l1": 0.0,
                "edge_smoothness": 0.0,
                "supervised_log_l1": 0.0,
                "supervised_silog": 0.0,
                "supervised_disp_l1": 0.0,
                "supervised_disp_log_l1": 0.0,
            },
            "photometric": {"alpha": 0.0},
        },
        "train": {"batch_size": 1},
    }

    metrics = run_dry_run(
        config,
        train_manifest=manifest,
        val_manifest=manifest,
        device=torch.device("cpu"),
        output_dir=tmp_path / "out",
        prior_builder=lambda _config: TinyPrior(),
    )

    assert metrics["adapter_grad_norm"] > 0.0
    assert (tmp_path / "out" / "panels" / "dry_run.png").exists()
