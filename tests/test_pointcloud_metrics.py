from __future__ import annotations

import numpy as np

from eaglevision.geometry.pointcloud_eval import pointcloud_metrics


def test_pointcloud_metrics_identical_clouds_are_perfect():
    points = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]], dtype=np.float32)

    metrics = pointcloud_metrics(points, points.copy())

    assert metrics["chamfer_l1"] == 0.0
    assert metrics["accuracy"] == 0.0
    assert metrics["completeness"] == 0.0
    assert metrics["fscore_0_02"] == 1.0
