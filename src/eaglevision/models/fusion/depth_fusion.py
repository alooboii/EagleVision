from __future__ import annotations

import torch


def fuse_warped_and_predicted_depth(
    warped_depth: torch.Tensor,
    predicted_depth: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Fill warp holes with predicted depth."""
    valid = valid_mask.to(warped_depth.dtype)
    return valid * warped_depth + (1.0 - valid) * predicted_depth
