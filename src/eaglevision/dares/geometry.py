from __future__ import annotations

import torch
import torch.nn.functional as F


def depth_to_disparity(
    depth: torch.Tensor,
    focal_length: torch.Tensor | float,
    baseline: torch.Tensor | float,
    min_depth: float = 1e-3,
) -> torch.Tensor:
    if depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth.squeeze(1)
    focal = torch.as_tensor(focal_length, device=depth.device, dtype=depth.dtype)
    base = torch.as_tensor(baseline, device=depth.device, dtype=depth.dtype)
    while focal.ndim < depth.ndim:
        focal = focal.unsqueeze(-1)
    while base.ndim < depth.ndim:
        base = base.unsqueeze(-1)
    return focal * base / depth.clamp_min(min_depth)


def _base_grid(batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    grid = torch.stack((2.0 * xs / max(width - 1, 1) - 1.0, 2.0 * ys / max(height - 1, 1) - 1.0), dim=-1)
    return grid.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()


def disparity_warp(image: torch.Tensor, disparity: torch.Tensor, direction: str = "left_to_right") -> dict[str, torch.Tensor]:
    if direction not in {"left_to_right", "right_to_left"}:
        raise ValueError("direction must be 'left_to_right' or 'right_to_left'")
    if disparity.ndim == 4 and disparity.shape[1] == 1:
        disparity = disparity.squeeze(1)
    batch_size, _, height, width = image.shape
    if disparity.shape != (batch_size, height, width):
        disparity = F.interpolate(disparity.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False).squeeze(1)
    grid = _base_grid(batch_size, height, width, image.device, image.dtype)
    shift = disparity.to(image.dtype)
    if direction == "right_to_left":
        shift = -shift
    grid = grid.clone()
    grid[..., 0] = grid[..., 0] + 2.0 * shift / max(width - 1, 1)
    warped = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    valid = torch.isfinite(disparity) & (grid[..., 0] >= -1.0) & (grid[..., 0] <= 1.0) & (grid[..., 1] >= -1.0) & (grid[..., 1] <= 1.0)
    return {"warped": warped, "valid_mask": valid, "grid": grid}
