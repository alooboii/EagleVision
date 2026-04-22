from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from eaglevision.utils.intrinsics import scale_intrinsics


@dataclass(frozen=True)
class ResizeResult:
    image: np.ndarray
    depth: np.ndarray | None
    intrinsics: np.ndarray


def resize_sample(
    image: np.ndarray,
    depth: np.ndarray | None,
    intrinsics: np.ndarray,
    size: tuple[int, int],
) -> ResizeResult:
    """Resize RGB and depth consistently while updating intrinsics."""
    dst_h, dst_w = size
    src_h, src_w = image.shape[:2]
    resized_image = cv2.resize(image, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
    resized_depth = None
    if depth is not None:
        resized_depth = cv2.resize(depth, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)
    k = torch.from_numpy(intrinsics).float()
    scaled_k = scale_intrinsics(k, (src_h, src_w), (dst_h, dst_w)).numpy()
    return ResizeResult(image=resized_image, depth=resized_depth, intrinsics=scaled_k)


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert HWC uint8 RGB into CHW float tensor in [0, 1]."""
    tensor = torch.from_numpy(np.ascontiguousarray(image)).float() / 255.0
    return tensor.permute(2, 0, 1)


def depth_to_tensor(depth: np.ndarray) -> torch.Tensor:
    """Convert a depth array into a float tensor."""
    return torch.from_numpy(np.ascontiguousarray(depth)).float()
