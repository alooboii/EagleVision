from __future__ import annotations

import torch


def make_pixel_grid(batch: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    """Create a batched homogeneous pixel grid with shape [B, 3, H, W]."""
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    grid = torch.stack((xs, ys, ones), dim=0).unsqueeze(0)
    return grid.repeat(batch, 1, 1, 1)


def backproject(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Backproject depth maps into camera-space 3D points [B, 3, H, W]."""
    batch, height, width = depth.shape
    grid = make_pixel_grid(batch, height, width, depth.device).reshape(batch, 3, -1)
    inv_k = torch.inverse(intrinsics)
    rays = inv_k @ grid
    points = rays * depth.reshape(batch, 1, -1)
    return points.reshape(batch, 3, height, width)


def transform_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Apply a homogeneous rigid transform to 3D points."""
    batch, _, height, width = points.shape
    flat = points.reshape(batch, 3, -1)
    ones = torch.ones((batch, 1, flat.shape[-1]), dtype=points.dtype, device=points.device)
    homogeneous = torch.cat((flat, ones), dim=1)
    transformed = transform @ homogeneous
    return transformed[:, :3].reshape(batch, 3, height, width)


def project(points: torch.Tensor, intrinsics: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Project camera-space 3D points into image pixels and return depth."""
    batch, _, height, width = points.shape
    flat = points.reshape(batch, 3, -1)
    depths = flat[:, 2].clamp_min(eps)
    pixels = intrinsics @ flat
    pixels_xy = pixels[:, :2] / depths.unsqueeze(1)
    return pixels_xy.reshape(batch, 2, height, width), depths.reshape(batch, height, width)
