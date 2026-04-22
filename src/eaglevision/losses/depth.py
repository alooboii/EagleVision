from __future__ import annotations

import torch

from eaglevision.utils.masks import masked_mean, valid_depth_mask


def masked_depth_l1(prediction: torch.Tensor, target: torch.Tensor, extra_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute masked depth L1 over valid target pixels."""
    mask = valid_depth_mask(target)
    if extra_mask is not None:
        mask = mask & extra_mask
    return masked_mean(torch.abs(prediction - target), mask)
