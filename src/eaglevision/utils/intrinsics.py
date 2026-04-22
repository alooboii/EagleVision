from __future__ import annotations

import torch


def scale_intrinsics(intrinsics: torch.Tensor, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> torch.Tensor:
    """Scale pinhole intrinsics after image resizing."""
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    scaled = intrinsics.clone()
    scaled[..., 0, 0] *= dst_w / src_w
    scaled[..., 1, 1] *= dst_h / src_h
    scaled[..., 0, 2] *= dst_w / src_w
    scaled[..., 1, 2] *= dst_h / src_h
    return scaled
