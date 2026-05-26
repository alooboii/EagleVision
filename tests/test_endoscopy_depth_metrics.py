from __future__ import annotations

import math

import torch

from eaglevision.metrics.depth_metrics import abs_rel, compute_depth_metrics, delta1, rmse


def test_depth_metrics_simple_tensors():
    pred = torch.tensor([[[1.0, 2.0], [4.0, 8.0]]])
    target = torch.tensor([[[1.0, 1.0], [2.0, 4.0]]])

    metrics = compute_depth_metrics(pred, target)

    assert abs_rel(pred, target) == 0.75
    assert rmse(pred, target) > 0
    assert metrics["sqrel"] > 0
    assert delta1(pred, target) == 0.25


def test_depth_metrics_return_nan_without_valid_pixels():
    pred = torch.zeros(1, 2, 2)
    target = torch.zeros(1, 2, 2)

    assert math.isnan(abs_rel(pred, target))
    assert math.isnan(compute_depth_metrics(pred, target)["rmse"])
