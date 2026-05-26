from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def load_depth_npy_or_png(path: str | Path, depth_scale: float = 1000.0) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32)
    return (np.asarray(Image.open(path), dtype=np.float32) / float(depth_scale)).astype(np.float32)


def load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_pose(path: str | Path) -> np.ndarray | None:
    try:
        pose = np.loadtxt(path, dtype=np.float32).reshape(4, 4)
    except Exception:
        return None
    return pose if np.isfinite(pose).all() else None


def load_intrinsics(path: str | Path) -> np.ndarray | None:
    try:
        mat = np.loadtxt(path, dtype=np.float32)
    except Exception:
        return None
    if mat.shape == (4, 4):
        mat = mat[:3, :3]
    return mat.reshape(3, 3) if mat.size >= 9 and np.isfinite(mat).all() else None


def backproject_depth(depth: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    z = depth.astype(np.float32)
    valid = np.isfinite(z) & (z > 0)
    x = (xs - K[0, 2]) * z / K[0, 0]
    y = (ys - K[1, 2]) * z / K[1, 1]
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return points[valid.reshape(-1)], valid


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.reshape(0, 3)
    homog = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    return (T @ homog.T).T[:, :3]


def project_points(points: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1e-6)
    pixels = np.zeros((points.shape[0], 2), dtype=np.float32)
    pixels[valid, 0] = K[0, 0] * points[valid, 0] / z[valid] + K[0, 2]
    pixels[valid, 1] = K[1, 1] * points[valid, 1] / z[valid] + K[1, 2]
    return pixels, z.astype(np.float32), valid


def project_depth_to_target(
    source_depth: np.ndarray,
    source_pose: np.ndarray,
    target_pose: np.ndarray,
    K_source: np.ndarray,
    K_target: np.ndarray,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    points_source, valid_source = backproject_depth(source_depth, K_source)
    if points_source.size == 0:
        return np.zeros(target_shape, dtype=np.float32), np.zeros(target_shape, dtype=bool)
    T_target_from_source = np.linalg.inv(target_pose) @ source_pose
    points_target = transform_points(points_source, T_target_from_source.astype(np.float32))
    pixels, z, valid_proj = project_points(points_target, K_target)
    h, w = target_shape
    depth_out = np.full((h, w), np.inf, dtype=np.float32)
    xi = np.rint(pixels[:, 0]).astype(np.int32)
    yi = np.rint(pixels[:, 1]).astype(np.int32)
    inside = valid_proj & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    for x, y, zz in zip(xi[inside], yi[inside], z[inside], strict=False):
        if zz < depth_out[y, x]:
            depth_out[y, x] = zz
    valid = np.isfinite(depth_out)
    depth_out[~valid] = 0.0
    return depth_out, valid


def warp_rgb_to_target(
    source_rgb: np.ndarray,
    source_depth: np.ndarray,
    source_pose: np.ndarray,
    target_pose: np.ndarray,
    K_source: np.ndarray,
    K_target: np.ndarray,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    points_source, valid_source = backproject_depth(source_depth, K_source)
    if points_source.size == 0:
        return np.zeros((*target_shape, 3), dtype=np.float32), np.zeros(target_shape, dtype=bool)
    colors = source_rgb.reshape(-1, 3)[valid_source.reshape(-1)]
    T_target_from_source = np.linalg.inv(target_pose) @ source_pose
    points_target = transform_points(points_source, T_target_from_source.astype(np.float32))
    pixels, z, valid_proj = project_points(points_target, K_target)
    h, w = target_shape
    zbuf = np.full((h, w), np.inf, dtype=np.float32)
    rgb_out = np.zeros((h, w, 3), dtype=np.float32)
    xi = np.rint(pixels[:, 0]).astype(np.int32)
    yi = np.rint(pixels[:, 1]).astype(np.int32)
    inside = valid_proj & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    for x, y, zz, color in zip(xi[inside], yi[inside], z[inside], colors[inside], strict=False):
        if zz < zbuf[y, x]:
            zbuf[y, x] = zz
            rgb_out[y, x] = color
    valid = np.isfinite(zbuf)
    return rgb_out, valid


def depth_errors(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    valid = np.isfinite(pred) & np.isfinite(target) & (pred > 0) & (target > 0)
    if mask is not None:
        valid &= mask
    if not valid.any():
        return {"depth_l1": float("nan"), "absrel": float("nan"), "rmse": float("nan"), "delta1": float("nan"), "num_valid_pixels": 0}
    p = pred[valid]
    t = target[valid]
    ratio = np.maximum(p / np.maximum(t, 1e-6), t / np.maximum(p, 1e-6))
    return {
        "depth_l1": float(np.mean(np.abs(p - t))),
        "absrel": float(np.mean(np.abs(p - t) / np.maximum(t, 1e-6))),
        "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
        "num_valid_pixels": int(valid.sum()),
    }


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path
