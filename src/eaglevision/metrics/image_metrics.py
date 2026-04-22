from __future__ import annotations

import math

import torch

try:
    from skimage.metrics import structural_similarity
except ModuleNotFoundError:  # pragma: no cover - dependency-dependent fallback
    structural_similarity = None

from eaglevision.losses.photometric import masked_l1


def rgb_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked RGB L1."""
    return masked_l1(prediction, target, mask)


def psnr(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Compute PSNR over valid RGB pixels."""
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    valid = mask.expand_as(prediction)
    mse = ((prediction - target) ** 2)[valid].mean().item() if valid.any() else 0.0
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(1.0 / math.sqrt(mse)))


def ssim(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Compute average SSIM on the first batch item for debugging-scale evaluation."""
    if structural_similarity is None:
        return 0.0
    pred = prediction[0].detach().cpu().permute(1, 2, 0).numpy()
    tgt = target[0].detach().cpu().permute(1, 2, 0).numpy()
    return float(structural_similarity(pred, tgt, channel_axis=2, data_range=1.0))
