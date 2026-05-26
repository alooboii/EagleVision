from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from eaglevision.dares.depth_anything_dares import DARESDepthAnything


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.ModuleDict({"attn": nn.ModuleDict({"qkv": nn.Linear(3, 3), "proj": nn.Linear(3, 3)})})])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pixels = x.permute(0, 2, 3, 1)
        y = self.blocks[0]["attn"]["proj"](self.blocks[0]["attn"]["qkv"](pixels))
        return F.softplus(y.mean(dim=-1)) + 0.1


def _config() -> dict:
    return {
        "base_model": {"normalize_backbone_input": False},
        "vector_lora": {"target_patterns": ["qkv", "proj"], "early_rank": 2, "default_rank": 2, "default_alpha": 4},
    }


def test_dares_depth_wrapper_forward_shape_padding_and_trainable_params():
    model = DARESDepthAnything(_config(), backbone=FakeBackbone())
    image = torch.rand(2, 3, 15, 17)
    depth = model(image)
    assert depth.shape == (2, 15, 17)
    assert model.vector_lora_summary()["replaced_modules"]
    trainable = model.trainable_parameter_names()
    assert trainable
    assert all(".lora_A." in name or ".lora_B." in name for name in trainable)

    loss = depth.mean()
    loss.backward()
    assert any(parameter.grad is not None for name, parameter in model.named_parameters() if ".lora_" in name)
