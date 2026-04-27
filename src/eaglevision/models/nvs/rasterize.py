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
    valid = torch.zeros((batch, height * width), dtype=torch.bool, device=flat_depth.device)

    # This avoids slow Python-per-point loops and keeps z-buffer logic explicit:
    # sort by depth then pixel-id, and keep the first (nearest) point per pixel.
    for batch_idx in range(batch):
        linear_idx = flat_linear_index[batch_idx]
        depth = flat_depth[batch_idx]
        rgb = flat_rgb[batch_idx]

        point_valid = linear_idx >= 0
        if not torch.any(point_valid):
            continue

        linear_idx = linear_idx[point_valid]
        depth = depth[point_valid]
        rgb = rgb[:, point_valid]

        depth_order = torch.argsort(depth, stable=True)
        linear_idx = linear_idx[depth_order]
        depth = depth[depth_order]
        rgb = rgb[:, depth_order]

        pixel_order = torch.argsort(linear_idx, stable=True)
        linear_idx = linear_idx[pixel_order]
        depth = depth[pixel_order]
        rgb = rgb[:, pixel_order]

        keep_first = torch.ones(linear_idx.shape[0], dtype=torch.bool, device=linear_idx.device)
        if keep_first.numel() > 1:
            keep_first[1:] = linear_idx[1:] != linear_idx[:-1]

        selected_idx = linear_idx[keep_first]
        selected_depth = depth[keep_first]
        selected_rgb = rgb[:, keep_first]

        out_depth[batch_idx, selected_idx] = selected_depth
        out_rgb[batch_idx, :, selected_idx] = selected_rgb
        valid[batch_idx, selected_idx] = True

    out_depth = torch.where(valid, out_depth, torch.zeros_like(out_depth))
    return (
        out_rgb.reshape(batch, channels, height, width),
        out_depth.reshape(batch, height, width),
        valid.reshape(batch, height, width),
    )
