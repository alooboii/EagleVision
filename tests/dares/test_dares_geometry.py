from __future__ import annotations

import torch

from eaglevision.dares.geometry import depth_to_disparity, disparity_warp


def test_zero_disparity_returns_same_image_and_valid_mask():
    image = torch.rand(1, 3, 4, 5)
    out = disparity_warp(image, torch.zeros(1, 4, 5), direction="left_to_right")
    assert torch.allclose(out["warped"], image, atol=1e-6)
    assert torch.all(out["valid_mask"])


def test_positive_disparity_shifts_and_invalidates_bounds():
    image = torch.arange(5, dtype=torch.float32).view(1, 1, 1, 5).expand(1, 1, 3, 5)
    out = disparity_warp(image, torch.ones(1, 3, 5), direction="left_to_right")
    assert torch.allclose(out["warped"][0, 0, :, 0], image[0, 0, :, 1])
    assert not torch.all(out["valid_mask"])


def test_warp_gradients_flow_to_disparity_and_depth_to_disparity():
    image = torch.rand(1, 3, 4, 5)
    disparity = torch.zeros(1, 4, 5, requires_grad=True)
    loss = disparity_warp(image, disparity)["warped"].mean()
    loss.backward()
    assert disparity.grad is not None

    depth = torch.full((2, 3, 4), 2.0)
    disp = depth_to_disparity(depth, torch.tensor([10.0, 20.0]), 0.5)
    assert torch.allclose(disp[0], torch.full((3, 4), 2.5))
    assert torch.allclose(disp[1], torch.full((3, 4), 5.0))
