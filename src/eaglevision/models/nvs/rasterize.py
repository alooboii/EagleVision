from __future__ import annotations

import torch


def z_buffer_scatter(
    flat_rgb: torch.Tensor,
    flat_depth: torch.Tensor,
    flat_linear_index: torch.Tensor,
    output_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scatter projected samples into an image using nearest-depth selection."""
    batch, channels, num_points = flat_rgb.shape
    height, width = output_hw
    out_rgb = torch.zeros((batch, channels, height * width), dtype=flat_rgb.dtype, device=flat_rgb.device)
    out_depth = torch.full((batch, height * width), float("inf"), dtype=flat_depth.dtype, device=flat_depth.device)
    hits = torch.zeros((batch, height * width), dtype=torch.int32, device=flat_depth.device)

    for batch_idx in range(batch):
        for point_idx in range(num_points):
            linear_idx = int(flat_linear_index[batch_idx, point_idx].item())
            depth = flat_depth[batch_idx, point_idx]
            if linear_idx < 0:
                continue
            hits[batch_idx, linear_idx] += 1
            if depth < out_depth[batch_idx, linear_idx]:
                out_depth[batch_idx, linear_idx] = depth
                out_rgb[batch_idx, :, linear_idx] = flat_rgb[batch_idx, :, point_idx]

    valid = torch.isfinite(out_depth)
    out_depth = torch.where(valid, out_depth, torch.zeros_like(out_depth))
    return (
        out_rgb.reshape(batch, channels, height, width),
        out_depth.reshape(batch, height, width),
        valid.reshape(batch, height, width),
    )
