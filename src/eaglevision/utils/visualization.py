from __future__ import annotations

import numpy as np
import torch


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a CHW float image into uint8 HWC RGB."""
    array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (array * 255.0).astype(np.uint8)


def depth_to_colormap(depth: torch.Tensor) -> np.ndarray:
    """Normalize a depth map for preview."""
    array = depth.detach().cpu().numpy()
    valid = np.isfinite(array) & (array > 0)
    if not valid.any():
        return np.zeros((*array.shape, 3), dtype=np.uint8)
    norm = np.zeros_like(array, dtype=np.float32)
    min_val = array[valid].min()
    max_val = array[valid].max()
    norm[valid] = (array[valid] - min_val) / max(max_val - min_val, 1e-6)
    rgb = np.stack([norm, norm, norm], axis=-1)
    return (rgb * 255.0).astype(np.uint8)
