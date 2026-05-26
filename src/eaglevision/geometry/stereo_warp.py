from __future__ import annotations

import torch
import torch.nn.functional as F


def make_normalized_grid(batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    x_norm = 2.0 * xs / max(width - 1, 1) - 1.0
    y_norm = 2.0 * ys / max(height - 1, 1) - 1.0
    grid = torch.stack((x_norm, y_norm), dim=-1)
    return grid.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()


def compute_disparity_from_depth(
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


def warp_image_with_disparity(
    image: torch.Tensor,
    disparity: torch.Tensor,
    direction: str = "left_to_right",
    sign: float = 1.0,
    padding_mode: str = "zeros",
) -> dict[str, torch.Tensor]:
    """Differentiably sample a stereo image using horizontal disparity.

    The default assumes conventional positive disparity d = x_left - x_right.
    For a left image reconstructed in the right view, grid_sample reads from
    source x_left = x_right + d.
    """
    if direction not in {"left_to_right", "right_to_left"}:
        raise ValueError("direction must be 'left_to_right' or 'right_to_left'")
    if disparity.ndim == 4 and disparity.shape[1] == 1:
        disparity = disparity.squeeze(1)
    if disparity.ndim != 3:
        raise ValueError(f"Expected disparity shape [B,H,W] or [B,1,H,W], got {tuple(disparity.shape)}")

    batch_size, _, height, width = image.shape
    if disparity.shape != (batch_size, height, width):
        disparity = F.interpolate(disparity.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False).squeeze(1)

    base_grid = make_normalized_grid(batch_size, height, width, image.device, image.dtype)
    pixel_shift = sign * disparity.to(image.dtype)
    if direction == "right_to_left":
        pixel_shift = -pixel_shift
    norm_shift = 2.0 * pixel_shift / max(width - 1, 1)
    grid = base_grid.clone()
    grid[..., 0] = grid[..., 0] + norm_shift

    warped = F.grid_sample(image, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True)
    valid = torch.isfinite(disparity) & (grid[..., 0] >= -1.0) & (grid[..., 0] <= 1.0) & (grid[..., 1] >= -1.0) & (grid[..., 1] <= 1.0)
    return {"warped": warped, "valid_mask": valid, "grid": grid}


def warp_image_with_depth_calibration(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
    raise NotImplementedError(
        "Depth+K+T stereo warping is reserved for the next iteration. "
        "Use disparity_gt or focal_length+baseline to train this first experiment path."
    )
