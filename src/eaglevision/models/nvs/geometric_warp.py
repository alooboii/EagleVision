from __future__ import annotations

import torch
from torch import nn

from eaglevision.models.nvs.rasterize import z_buffer_scatter
from eaglevision.models.nvs.visibility import compute_projection_mask
from eaglevision.utils.geometry import backproject, project, transform_points
from eaglevision.utils.masks import valid_depth_mask


class GeometricWarper(nn.Module):
    """Explicit RGB-D forward projector with simple z-buffer visibility."""

    def forward(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        src_intrinsics: torch.Tensor,
        dst_intrinsics: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, _, height, width = image.shape
        depth_valid = valid_depth_mask(depth)
        points_src = backproject(depth, src_intrinsics)
        points_dst = transform_points(points_src, src_to_dst)
        pixels_dst, depth_dst = project(points_dst, dst_intrinsics)

        flat_pixels = pixels_dst.reshape(batch, 2, -1)
        flat_depth = depth_dst.reshape(batch, -1)
        flat_valid = depth_valid.reshape(batch, -1)
        valid_proj, linear_idx = compute_projection_mask(flat_pixels, flat_depth, (height, width))
        flat_rgb = image.reshape(batch, image.shape[1], -1)
        linear_idx = torch.where(flat_valid & valid_proj, linear_idx, torch.full_like(linear_idx, -1))

        warped_rgb, warped_depth, valid_mask = z_buffer_scatter(flat_rgb, flat_depth, linear_idx, (height, width))
        hole_mask = ~valid_mask
        return {
            "warped_rgb": warped_rgb,
            "warped_depth": warped_depth,
            "valid_mask": valid_mask,
            "hole_mask": hole_mask,
        }
