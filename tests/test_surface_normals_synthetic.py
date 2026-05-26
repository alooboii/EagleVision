from __future__ import annotations

import numpy as np

from eaglevision.cli.eval_surface_normals import normal_metrics


def test_surface_normals_identical_depth_has_zero_error():
    depth = np.ones((6, 6), dtype=np.float32) * 2.0
    K = np.array([[10.0, 0.0, 3.0], [0.0, 10.0, 3.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    metrics = normal_metrics(depth, depth.copy(), K)

    assert metrics["num_valid_pixels"] > 0
    assert metrics["normal_mae_deg"] < 1e-4
    assert metrics["normal_11_25"] == 1.0
