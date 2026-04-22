from __future__ import annotations

import torch


def valid_depth_mask(depth: torch.Tensor) -> torch.Tensor:
    """Return a boolean validity mask for depth tensors."""
    return torch.isfinite(depth) & (depth > 0)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute the mean over valid elements only."""
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(eps)
