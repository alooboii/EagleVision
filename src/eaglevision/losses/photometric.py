from __future__ import annotations

import torch

from eaglevision.utils.masks import masked_mean


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute masked image L1."""
    if mask.ndim == prediction.ndim - 1:
        mask = mask.unsqueeze(1)
    return masked_mean(torch.abs(prediction - target), mask.expand_as(prediction))
