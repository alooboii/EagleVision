from __future__ import annotations

import torch

from eaglevision.models.residual_log_depth_adapter import ResidualLogDepthAdapter


def test_residual_log_depth_adapter_shapes_and_positive_depth():
    adapter = ResidualLogDepthAdapter(hidden_channels=8)
    image = torch.rand(2, 3, 5, 7)
    base_depth = torch.ones(2, 5, 7)

    outputs = adapter(image, base_depth)

    assert outputs["adapted_depth"].shape == (2, 5, 7)
    assert outputs["base_depth"].shape == (2, 5, 7)
    assert outputs["residual"].shape == (2, 1, 5, 7)
    assert torch.all(outputs["adapted_depth"] > 0)
