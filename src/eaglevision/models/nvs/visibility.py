from __future__ import annotations

import torch


def compute_projection_mask(
    pixels: torch.Tensor,
    depth: torch.Tensor,
    output_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute valid projection masks and flattened target indices."""
    height, width = output_hw
    x = pixels[:, 0].round().long()
    y = pixels[:, 1].round().long()
    in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    front = depth > 0
    valid = in_bounds & front & torch.isfinite(depth)
    linear_idx = torch.where(valid, y * width + x, torch.full_like(x, -1))
    return valid, linear_idx
