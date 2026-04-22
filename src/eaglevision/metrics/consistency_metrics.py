from __future__ import annotations

import torch

from eaglevision.utils.masks import valid_depth_mask


def depth_reprojection_consistency(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean reprojection depth error on valid pixels."""
    valid = valid_depth_mask(target) & mask
    return float(torch.abs(prediction[valid] - target[valid]).mean().item()) if valid.any() else 0.0
