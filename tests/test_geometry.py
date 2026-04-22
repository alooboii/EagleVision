from __future__ import annotations

import torch

from eaglevision.utils.geometry import backproject, project, transform_points


def test_backproject_and_project_preserve_shape() -> None:
    depth = torch.ones((2, 4, 5))
    intrinsics = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
    points = backproject(depth, intrinsics)
    assert points.shape == (2, 3, 4, 5)
    transformed = transform_points(points, torch.eye(4).unsqueeze(0).repeat(2, 1, 1))
    pixels, projected_depth = project(transformed, intrinsics)
    assert pixels.shape == (2, 2, 4, 5)
    assert projected_depth.shape == (2, 4, 5)
