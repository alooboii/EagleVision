from __future__ import annotations

import pytest
import torch

from eaglevision.dares.losses import compute_dares_losses, multi_scale_ssim_reprojection_loss


def _batch(disparity_value: float = 99.0) -> dict[str, torch.Tensor]:
    xs = torch.linspace(0, 1, 7).view(1, 1, 1, 7).expand(1, 3, 5, 7)
    left = xs.square()
    return {
        "left_rgb": left,
        "right_rgb": torch.roll(left, shifts=-1, dims=-1),
        "focal_length": torch.tensor([2.0]),
        "baseline": torch.tensor([1.0]),
        "valid_mask": torch.ones(1, 5, 7, dtype=torch.bool),
        "disparity_gt": torch.full((1, 5, 7), disparity_value),
    }


def _config() -> dict:
    return {
        "reprojection": {"scales": [1, 2], "alpha": 0.85, "symmetric": False},
        "weights": {"reproj": 1.0, "smooth": 0.0, "supervised_depth": 0.0, "supervised_disparity": 0.0},
    }


def test_multi_scale_ssim_finite_with_mask():
    image = torch.rand(1, 3, 6, 8)
    loss = multi_scale_ssim_reprojection_loss(image, image, torch.ones(1, 6, 8, dtype=torch.bool), scales=(1, 2))
    assert torch.isfinite(loss)


def test_dares_loss_changes_with_predicted_depth_and_has_gradients():
    batch = _batch()
    depth_near = torch.ones(1, 5, 7, requires_grad=True)
    depth_far = torch.full((1, 5, 7), 4.0, requires_grad=True)
    loss_near = compute_dares_losses(batch, {"pred_depth_left": depth_near}, _config())["loss_total"]
    loss_far = compute_dares_losses(batch, {"pred_depth_left": depth_far}, _config())["loss_total"]
    assert not torch.allclose(loss_near, loss_far)
    loss_near.backward()
    assert depth_near.grad is not None
    assert float(depth_near.grad.abs().sum()) > 0


def test_dares_reprojection_ignores_disparity_gt_and_requires_geometry():
    depth = torch.ones(1, 5, 7)
    loss_a = compute_dares_losses(_batch(disparity_value=1.0), {"pred_depth_left": depth}, _config())["loss_reproj_ms_ssim"]
    loss_b = compute_dares_losses(_batch(disparity_value=100.0), {"pred_depth_left": depth}, _config())["loss_reproj_ms_ssim"]
    assert torch.allclose(loss_a, loss_b)
    bad = _batch()
    bad.pop("focal_length")
    with pytest.raises(ValueError, match="focal_length and baseline"):
        compute_dares_losses(bad, {"pred_depth_left": depth}, _config())
