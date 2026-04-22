from __future__ import annotations

import torch

from eaglevision.utils.masks import valid_depth_mask


def depth_l1(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Masked depth L1."""
    mask = valid_depth_mask(target)
    return float(torch.abs(prediction[mask] - target[mask]).mean().item()) if mask.any() else 0.0


def rmse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Masked depth RMSE."""
    mask = valid_depth_mask(target)
    return float(torch.sqrt(((prediction[mask] - target[mask]) ** 2).mean()).item()) if mask.any() else 0.0


def abs_rel(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Masked absolute relative error."""
    mask = valid_depth_mask(target)
    return float((torch.abs(prediction[mask] - target[mask]) / target[mask].clamp_min(1e-6)).mean().item()) if mask.any() else 0.0
