from __future__ import annotations

import csv
import json

import numpy as np
import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F

from eaglevision.dares import evaluator as dares_evaluator
from eaglevision.dares.metrics import reprojection_metrics
from eaglevision.dares.vector_lora import apply_vector_lora, vector_lora_state_dict


class FakeDARESModel(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.blocks = nn.ModuleList([nn.ModuleDict({"attn": nn.ModuleDict({"qkv": nn.Linear(3, 3), "proj": nn.Linear(3, 1)})})])
        self._summary = apply_vector_lora(self.backbone, ["qkv", "proj"], {"default_rank": 2, "default_alpha": 4}, dropout=0.0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pixels = image.permute(0, 2, 3, 1)
        y = self.backbone.blocks[0]["attn"]["proj"](self.backbone.blocks[0]["attn"]["qkv"](pixels)).squeeze(-1)
        return F.softplus(y) + 0.5

    def predict_depth(self, image: torch.Tensor) -> torch.Tensor:
        return self.forward(image)

    def vector_lora_summary(self):
        return self._summary

    def trainable_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def _write_rgb(path) -> None:
    image = np.tile(np.linspace(0, 255, 7, dtype=np.uint8)[None, :, None], (5, 1, 3))
    Image.fromarray(image).save(path)


def test_eval_dares_writes_csv_and_summary(monkeypatch, tmp_path):
    _write_rgb(tmp_path / "left.png")
    _write_rgb(tmp_path / "right.png")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"sample_id": "s0", "sequence_id": "seq", "frame_id": "0", "left_rgb": "left.png", "right_rgb": "right.png", "baseline": 1.0, "focal_length": 2.0}) + "\n",
        encoding="utf-8",
    )
    config = {
        "method": "dares_da2s_vector_lora",
        "device": "cpu",
        "data": {"root": str(tmp_path), "image_size": [5, 7]},
        "vector_lora": {"target_patterns": ["qkv", "proj"]},
        "loss": {"reprojection": {"scales": [1], "alpha": 0.85}},
        "eval": {"batch_size": 1, "num_workers": 0},
    }
    model = FakeDARESModel(config)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"vector_lora": vector_lora_state_dict(model)}, checkpoint)
    monkeypatch.setattr(dares_evaluator, "DARESDepthAnything", FakeDARESModel)

    out = tmp_path / "eval.csv"
    dares_evaluator.evaluate_dares(config, manifest, checkpoint, out)

    with out.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert (tmp_path / "eval.summary.json").exists()


def test_reprojection_metrics_change_with_prediction_depth():
    left = torch.linspace(0, 1, 7).view(1, 1, 1, 7).expand(1, 3, 5, 7).square()
    right = torch.roll(left, shifts=-1, dims=-1)
    near = reprojection_metrics(left, right, torch.ones(1, 5, 7), torch.tensor([2.0]), torch.tensor([1.0]), scales=(1,))
    far = reprojection_metrics(left, right, torch.full((1, 5, 7), 4.0), torch.tensor([2.0]), torch.tensor([1.0]), scales=(1,))
    assert near["reproj_ms_ssim"] != far["reproj_ms_ssim"]
