from __future__ import annotations

import torch

from eaglevision.losses.endoscopy_cycle_losses import compute_endoscopy_losses
from eaglevision.models.residual_log_depth_adapter import ResidualLogDepthAdapter


def _tiny_batch(include_disparity_gt: bool = True) -> dict[str, torch.Tensor]:
    xs = torch.linspace(0.0, 1.0, steps=7).view(1, 1, 1, 7).expand(1, 3, 5, 7)
    ys = torch.linspace(0.0, 1.0, steps=5).view(1, 1, 5, 1).expand(1, 3, 5, 7)
    left = (xs.square() + 0.25 * ys).clamp(0, 1)
    right = torch.roll(left, shifts=-1, dims=-1)
    batch = {
        "left_rgb": left,
        "right_rgb": right,
        "focal_length": torch.tensor([2.0]),
        "baseline": torch.tensor([1.0]),
        "valid_mask": torch.ones(1, 5, 7, dtype=torch.bool),
    }
    if include_disparity_gt:
        batch["disparity_gt"] = torch.full((1, 5, 7), 42.0)
    return batch


def _loss_config() -> dict:
    return {
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
    }


def _grad_norm(adapter: ResidualLogDepthAdapter) -> float:
    total = 0.0
    for parameter in adapter.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().abs().sum().item())
    return total


def test_photo_and_cycle_losses_backprop_to_adapter_parameters():
    torch.manual_seed(7)
    batch = _tiny_batch(include_disparity_gt=False)
    adapter = ResidualLogDepthAdapter(hidden_channels=4, residual_scale=0.4, use_tanh=False)
    base_depth = torch.ones(1, 5, 7)

    outputs = adapter(batch["left_rgb"], base_depth)
    losses = compute_endoscopy_losses(batch, outputs, _loss_config())
    adapter.zero_grad(set_to_none=True)
    losses["loss_photo"].backward(retain_graph=True)
    assert _grad_norm(adapter) > 0.0

    adapter.zero_grad(set_to_none=True)
    losses["loss_cycle_rgb"].backward()
    assert _grad_norm(adapter) > 0.0


def test_disparity_gt_does_not_decouple_photo_cycle_from_adapted_depth():
    batch = _tiny_batch(include_disparity_gt=True)
    base = torch.ones(1, 5, 7)
    outputs_near = {
        "base_depth": base,
        "adapted_depth": torch.ones(1, 5, 7),
        "log_base_depth": base.log(),
        "log_adapted_depth": torch.ones(1, 5, 7).log(),
        "residual": torch.zeros(1, 1, 5, 7),
    }
    outputs_far = {
        "base_depth": base,
        "adapted_depth": torch.full((1, 5, 7), 4.0),
        "log_base_depth": base.log(),
        "log_adapted_depth": torch.full((1, 5, 7), 4.0).log(),
        "residual": torch.zeros(1, 1, 5, 7),
    }

    losses_near = compute_endoscopy_losses(batch, outputs_near, _loss_config())
    losses_far = compute_endoscopy_losses(batch, outputs_far, _loss_config())

    assert not torch.allclose(losses_near["loss_photo"], losses_far["loss_photo"])
    assert not torch.allclose(losses_near["loss_cycle_rgb"], losses_far["loss_cycle_rgb"])
