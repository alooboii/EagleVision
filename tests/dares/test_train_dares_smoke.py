from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F

from eaglevision.dares import trainer as dares_trainer
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

    def vector_lora_summary(self):
        return self._summary

    def trainable_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def total_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def trainable_parameter_names(self):
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]


def _write_rgb(path) -> None:
    image = np.tile(np.linspace(0, 255, 7, dtype=np.uint8)[None, :, None], (5, 1, 3))
    Image.fromarray(image).save(path)


def test_train_dares_dry_run_writes_vector_lora_checkpoint(monkeypatch, tmp_path):
    _write_rgb(tmp_path / "left.png")
    _write_rgb(tmp_path / "right.png")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"sample_id": "s0", "left_rgb": "left.png", "right_rgb": "right.png", "baseline": 1.0, "focal_length": 2.0}) + "\n",
        encoding="utf-8",
    )
    config = {
        "seed": 42,
        "device": "cpu",
        "data": {"root": str(tmp_path), "image_size": [5, 7]},
        "vector_lora": {"target_patterns": ["qkv", "proj"]},
        "loss": {"reprojection": {"scales": [1], "symmetric": False}, "weights": {"reproj": 1.0, "smooth": 0.0}},
        "train": {"batch_size": 1},
        "output_dir": str(tmp_path / "out"),
    }
    monkeypatch.setattr(dares_trainer, "DARESDepthAnything", FakeDARESModel)
    metrics = dares_trainer.run_dry_run(config, manifest, manifest)
    assert metrics["vector_lora_grad_norm"] > 0
    checkpoint = torch.load(tmp_path / "out" / "checkpoints" / "dry_run.pt", map_location="cpu")
    assert "vector_lora" in checkpoint
    assert checkpoint["vector_lora"]
