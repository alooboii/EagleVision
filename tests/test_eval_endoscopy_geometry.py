from __future__ import annotations

import math

import torch

from eaglevision.cli.eval_endoscopy import _geometry_metrics


def test_geometry_metrics_use_prediction_depth_not_disparity_gt():
    xs = torch.linspace(0.0, 1.0, steps=7).view(1, 1, 1, 7).expand(1, 3, 5, 7)
    ys = torch.linspace(0.0, 1.0, steps=5).view(1, 1, 5, 1).expand(1, 3, 5, 7)
    left = (xs.square() + 0.25 * ys).clamp(0, 1)
    batch = {
        "left_rgb": left,
        "right_rgb": torch.roll(left, shifts=-1, dims=-1),
        "focal_length": torch.tensor([2.0]),
        "baseline": torch.tensor([1.0]),
        "disparity_gt": torch.full((1, 5, 7), 42.0),
    }
    config = {"data": {"min_depth": 1e-3}, "loss": {"photometric": {"alpha": 0.0}}}

    near = _geometry_metrics(batch, torch.ones(1, 5, 7), config)
    far = _geometry_metrics(batch, torch.full((1, 5, 7), 4.0), config)

    assert math.isfinite(near["photo_l1"])
    assert math.isfinite(far["photo_l1"])
    assert near["photo_l1"] != far["photo_l1"]
    assert near["cycle_rgb_l1"] != far["cycle_rgb_l1"]
    assert near["disp_rmse"] != far["disp_rmse"]
