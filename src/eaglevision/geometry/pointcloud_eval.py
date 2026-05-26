from __future__ import annotations

import numpy as np

from eaglevision.geometry.rgbd_reprojection import backproject_depth, transform_points


def depth_to_world_points(depth: np.ndarray, K: np.ndarray, pose: np.ndarray, max_points: int | None = None) -> np.ndarray:
    points_cam, _ = backproject_depth(depth, K)
    points_world = transform_points(points_cam, pose.astype(np.float32))
    if max_points is not None and points_world.shape[0] > max_points:
        indices = np.linspace(0, points_world.shape[0] - 1, max_points).astype(np.int64)
        points_world = points_world[indices]
    return points_world.astype(np.float32)


def _nearest_distances(a: np.ndarray, b: np.ndarray, chunk: int = 2048) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.full((a.shape[0],), np.nan, dtype=np.float32)
    out = []
    for start in range(0, a.shape[0], chunk):
        block = a[start : start + chunk]
        dist2 = np.sum((block[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        out.append(np.sqrt(np.min(dist2, axis=1)))
    return np.concatenate(out).astype(np.float32)


def pointcloud_metrics(pred_points: np.ndarray, ref_points: np.ndarray) -> dict[str, float]:
    if pred_points.size == 0 or ref_points.size == 0:
        return {
            "chamfer_l2": float("nan"),
            "chamfer_l1": float("nan"),
            "accuracy": float("nan"),
            "completeness": float("nan"),
            "fscore_0_02": float("nan"),
            "fscore_0_05": float("nan"),
            "fscore_0_10": float("nan"),
        }
    pred_to_ref = _nearest_distances(pred_points, ref_points)
    ref_to_pred = _nearest_distances(ref_points, pred_points)
    metrics = {
        "chamfer_l1": float(np.nanmean(pred_to_ref) + np.nanmean(ref_to_pred)),
        "chamfer_l2": float(np.nanmean(pred_to_ref**2) + np.nanmean(ref_to_pred**2)),
        "accuracy": float(np.nanmean(pred_to_ref)),
        "completeness": float(np.nanmean(ref_to_pred)),
    }
    for thresh in (0.02, 0.05, 0.10):
        precision = float(np.nanmean(pred_to_ref < thresh))
        recall = float(np.nanmean(ref_to_pred < thresh))
        metrics[f"fscore_{str(thresh).replace('.', '_')}"] = 2 * precision * recall / max(precision + recall, 1e-12)
    return metrics
