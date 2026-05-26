from __future__ import annotations

import numpy as np

from eaglevision.geometry.rgbd_reprojection import project_depth_to_target


def test_project_depth_identity_reprojects_same_depth():
    depth = np.ones((3, 4), dtype=np.float32) * 2.0
    K = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)

    projected, valid = project_depth_to_target(depth, pose, pose, K, K, depth.shape)

    assert valid.all()
    assert np.allclose(projected, depth)
