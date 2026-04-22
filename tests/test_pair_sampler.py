from __future__ import annotations

import numpy as np

from eaglevision.data.pair_sampler import PairSamplingConfig, filter_candidate_pairs


def _pose(tx: float, yaw_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pose = np.eye(4, dtype=np.float32)
    pose[0, 0] = np.cos(yaw)
    pose[0, 2] = np.sin(yaw)
    pose[2, 0] = -np.sin(yaw)
    pose[2, 2] = np.cos(yaw)
    pose[0, 3] = tx
    return pose


def test_pair_sampler_filters_by_motion_limits() -> None:
    poses = [_pose(0.0, 0.0), _pose(0.1, 5.0), _pose(0.6, 12.0)]
    pairs = filter_candidate_pairs(poses, [0, 1, 2], PairSamplingConfig())
    assert pairs == [(0, 1)]
