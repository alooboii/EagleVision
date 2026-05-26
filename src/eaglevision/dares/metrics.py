from __future__ import annotations

import torch

from eaglevision.dares.geometry import depth_to_disparity, disparity_warp
from eaglevision.dares.losses import multi_scale_ssim_reprojection_loss
from eaglevision.metrics.depth_metrics import compute_depth_metrics


def disparity_metrics(predicted_disparity: torch.Tensor, gt_disparity: torch.Tensor) -> dict[str, float]:
    pred = predicted_disparity.squeeze(1) if predicted_disparity.ndim == 4 and predicted_disparity.shape[1] == 1 else predicted_disparity
    target = gt_disparity.to(pred.device)
    target = target.squeeze(1) if target.ndim == 4 and target.shape[1] == 1 else target
    valid = torch.isfinite(pred) & torch.isfinite(target) & (pred > 0) & (target > 0)
    if not valid.any():
        return {"disp_absrel": float("nan"), "disp_rmse": float("nan"), "disp_log_l1": float("nan")}
    diff = pred[valid] - target[valid]
    log_diff = torch.log(pred[valid].clamp_min(1e-6)) - torch.log(target[valid].clamp_min(1e-6))
    return {
        "disp_absrel": float((diff.abs() / target[valid].clamp_min(1e-6)).mean().detach().cpu().item()),
        "disp_rmse": float(torch.sqrt((diff**2).mean()).detach().cpu().item()),
        "disp_log_l1": float(log_diff.abs().mean().detach().cpu().item()),
    }


def reprojection_metrics(
    left_rgb: torch.Tensor,
    right_rgb: torch.Tensor,
    predicted_depth: torch.Tensor,
    focal_length: torch.Tensor,
    baseline: torch.Tensor,
    scales: tuple[int, ...] = (1, 2, 4),
    alpha: float = 0.85,
    min_depth: float = 1e-3,
) -> dict[str, float | torch.Tensor]:
    disparity = depth_to_disparity(predicted_depth, focal_length, baseline, min_depth=min_depth)
    reproj = disparity_warp(left_rgb, disparity, direction="left_to_right")
    mask = reproj["valid_mask"]
    ms_ssim = multi_scale_ssim_reprojection_loss(reproj["warped"], right_rgb, mask, scales=scales, alpha=alpha)
    photo_l1 = torch.abs(reproj["warped"] - right_rgb).mean(dim=1)[mask].mean() if mask.any() else torch.tensor(float("nan"), device=left_rgb.device)
    return {
        "reproj_ms_ssim": float(ms_ssim.detach().cpu().item()),
        "photo_l1": float(photo_l1.detach().cpu().item()),
        "valid_reproj_ratio": float(mask.float().mean().detach().cpu().item()),
        "predicted_disparity": disparity,
    }


__all__ = ["compute_depth_metrics", "disparity_metrics", "reprojection_metrics"]
