from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn.functional as F

from eaglevision.geometry.stereo_warp import compute_disparity_from_depth, warp_image_with_disparity


_WARNED_NO_STEREO = False


def _as_bhw(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.squeeze(1) if tensor.ndim == 4 and tensor.shape[1] == 1 else tensor


def ssim_approx(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool2d(pred, 3, stride=1, padding=1)
    mu_y = F.avg_pool2d(target, 3, stride=1, padding=1)
    sigma_x = F.avg_pool2d(pred * pred, 3, stride=1, padding=1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, 3, stride=1, padding=1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, 3, stride=1, padding=1) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2))
    return ((1.0 - score.clamp(-1, 1)) * 0.5).clamp(0, 1)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    if mask.ndim == values.ndim - 1:
        mask = mask.unsqueeze(1)
    mask = mask.to(dtype=values.dtype, device=values.device).expand_as(values)
    denom = mask.sum()
    if denom <= 0:
        return values.new_zeros(())
    return (values * mask).sum() / denom.clamp_min(1.0)


def photometric_ssim_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None, alpha: float = 0.85) -> torch.Tensor:
    ssim_loss = ssim_approx(pred, target)
    l1_loss = torch.abs(pred - target)
    loss = alpha * ssim_loss + (1.0 - alpha) * l1_loss
    return _masked_mean(loss, mask)


def edge_aware_smoothness(depth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    depth = _as_bhw(depth)
    depth_norm = depth / depth.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    grad_depth_x = torch.abs(depth_norm[:, :, 1:] - depth_norm[:, :, :-1])
    grad_depth_y = torch.abs(depth_norm[:, 1:, :] - depth_norm[:, :-1, :])
    grad_img_x = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]), dim=1)
    grad_img_y = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]), dim=1)
    smooth_x = grad_depth_x * torch.exp(-grad_img_x)
    smooth_y = grad_depth_y * torch.exp(-grad_img_y)
    return smooth_x.mean() + smooth_y.mean()


def prior_log_l1(log_adapted: torch.Tensor, log_base: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    return _masked_mean(torch.abs(_as_bhw(log_adapted) - _as_bhw(log_base)), mask)


def supervised_depth_losses(adapted_depth: torch.Tensor, gt_depth: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    pred = _as_bhw(adapted_depth)
    target = _as_bhw(gt_depth)
    valid = torch.isfinite(pred) & torch.isfinite(target) & (target > 0) & (pred > 0)
    if mask is not None:
        valid = valid & mask.bool()
    if not valid.any():
        zero = pred.new_zeros(())
        return {"log_l1": zero, "silog": zero}
    log_diff = torch.log(pred[valid].clamp_min(1e-6)) - torch.log(target[valid].clamp_min(1e-6))
    log_l1 = log_diff.abs().mean()
    silog_var = ((log_diff**2).mean() - 0.85 * log_diff.mean() ** 2).clamp_min(0.0)
    silog = torch.sqrt(silog_var) * 100.0
    return {"log_l1": log_l1, "silog": silog}


def _stereo_disparity(batch: dict[str, Any], adapted_depth: torch.Tensor, min_depth: float) -> torch.Tensor | None:
    if batch.get("disparity_gt") is not None:
        return batch["disparity_gt"].to(adapted_depth.device)
    if batch.get("focal_length") is not None and batch.get("baseline") is not None:
        return compute_disparity_from_depth(
            adapted_depth,
            batch["focal_length"].to(adapted_depth.device),
            batch["baseline"].to(adapted_depth.device),
            min_depth=min_depth,
        )
    return None


def compute_endoscopy_losses(batch: dict[str, Any], outputs: dict[str, torch.Tensor], config: dict[str, Any]) -> dict[str, torch.Tensor]:
    global _WARNED_NO_STEREO

    weights = config.get("weights", {})
    alpha = float(config.get("photometric", {}).get("alpha", 0.85))
    left = batch["left_rgb"].to(outputs["adapted_depth"].device)
    right = batch["right_rgb"].to(outputs["adapted_depth"].device)
    adapted_depth = outputs["adapted_depth"]
    valid_mask = batch.get("valid_mask")
    if valid_mask is not None:
        valid_mask = valid_mask.to(adapted_depth.device).bool()

    zero = adapted_depth.new_zeros(())
    losses = {
        "loss_photo": zero,
        "loss_cycle_rgb": zero,
        "loss_prior_log_l1": prior_log_l1(outputs["log_adapted_depth"], outputs["log_base_depth"], valid_mask),
        "loss_edge_smoothness": edge_aware_smoothness(adapted_depth, left),
        "loss_supervised_log_l1": zero,
        "loss_supervised_silog": zero,
    }

    disparity = _stereo_disparity(batch, adapted_depth, min_depth=float(config.get("min_depth", 1e-3)))
    if disparity is None:
        if not _WARNED_NO_STEREO:
            warnings.warn("No disparity or focal_length+baseline in batch; photometric and cycle losses are zero.", stacklevel=2)
            _WARNED_NO_STEREO = True
    else:
        right_recon = warp_image_with_disparity(left, disparity, direction="left_to_right")
        mask = right_recon["valid_mask"]
        losses["loss_photo"] = photometric_ssim_l1(right_recon["warped"], right, mask=mask, alpha=alpha)

        left_cycle = warp_image_with_disparity(right_recon["warped"], disparity, direction="right_to_left")
        cycle_mask = mask & left_cycle["valid_mask"]
        losses["loss_cycle_rgb"] = _masked_mean(torch.abs(left_cycle["warped"] - left), cycle_mask)

    if batch.get("depth_gt") is not None:
        supervised = supervised_depth_losses(adapted_depth, batch["depth_gt"].to(adapted_depth.device), valid_mask)
        losses["loss_supervised_log_l1"] = supervised["log_l1"]
        losses["loss_supervised_silog"] = supervised["silog"]

    total = zero
    total = total + float(weights.get("photo", 1.0)) * losses["loss_photo"]
    total = total + float(weights.get("cycle_rgb", 0.5)) * losses["loss_cycle_rgb"]
    total = total + float(weights.get("prior_log_l1", 0.02)) * losses["loss_prior_log_l1"]
    total = total + float(weights.get("edge_smoothness", 0.001)) * losses["loss_edge_smoothness"]
    total = total + float(weights.get("supervised_log_l1", 0.0)) * losses["loss_supervised_log_l1"]
    total = total + float(weights.get("supervised_silog", 0.0)) * losses["loss_supervised_silog"]
    losses["loss_total"] = total
    return losses
