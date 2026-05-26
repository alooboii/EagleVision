from __future__ import annotations

import torch

from eaglevision.geometry.stereo_warp import compute_disparity_from_depth, warp_image_with_disparity


def test_zero_disparity_warp_is_identity():
    image = torch.rand(1, 3, 4, 5)
    disparity = torch.zeros(1, 4, 5)

    output = warp_image_with_disparity(image, disparity)

    assert output["warped"].shape == image.shape
    assert output["valid_mask"].shape == (1, 4, 5)
    assert torch.all(output["valid_mask"])
    assert torch.allclose(output["warped"], image, atol=1e-6)


def test_compute_disparity_from_depth():
    depth = torch.full((2, 3, 4), 2.0)
    disparity = compute_disparity_from_depth(depth, focal_length=torch.tensor([10.0, 20.0]), baseline=0.5)

    assert disparity.shape == depth.shape
    assert torch.allclose(disparity[0], torch.full((3, 4), 2.5))
    assert torch.allclose(disparity[1], torch.full((3, 4), 5.0))
