from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn.functional as F

from eaglevision.dares.geometry import depth_to_disparity, disparity_warp


_WARNED_EMPTY_MASK = False


def _as_bhw(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.squeeze(1) if tensor.ndim == 4 and tensor.shape[1] == 1 else tensor


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    if mask.ndim == values.ndim - 1:
        mask = mask.unsqueeze(1)
    mask = mask.to(values.device, values.dtype).expand_as(values)
    denom = mask.sum()
    if denom <= 0:
        global _WARNED_EMPTY_MASK
        if not _WARNED_EMPTY_MASK:
            warnings.warn("DARES reprojection loss received no valid pixels; returning zero.", stacklevel=2)
            _WARNED_EMPTY_MASK = True
        return values.new_zeros(())
    return (values * mask).sum() / denom.clamp_min(1.0)


def ssim_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool2d(pred, 3, stride=1, padding=1)
    mu_y = F.avg_pool2d(target, 3, stride=1, padding=1)
    sigma_x = F.avg_pool2d(pred * pred, 3, stride=1, padding=1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, 3, stride=1, padding=1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, 3, stride=1, padding=1) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2))
    return ((1.0 - score.clamp(-1, 1)) * 0.5).clamp(0, 1)


def multi_scale_ssim_reprojection_loss(
    pred_image: torch.Tensor,
    target_image: torch.Tensor,
    valid_mask: torch.Tensor | None,
    scales: tuple[int, ...] | list[int] = (1, 2, 4),
    alpha: float = 0.85,
) -> torch.Tensor:
    losses = []
    for scale in scales:
        if scale <= 1:
            pred_s = pred_image
            target_s = target_image
            mask_s = valid_mask
        else:
            pred_s = F.avg_pool2d(pred_image, kernel_size=scale, stride=scale)
            target_s = F.avg_pool2d(target_image, kernel_size=scale, stride=scale)
            if valid_mask is None:
                mask_s = None
            else:
                mask_s = F.avg_pool2d(valid_mask.unsqueeze(1).float(), kernel_size=scale, stride=scale).squeeze(1) > 0.5
        error = alpha * ssim_error(pred_s, target_s) + (1.0 - alpha) * torch.abs(pred_s - target_s)
        losses.append(_masked_mean(error, mask_s))
    return torch.stack(losses).mean()


def edge_aware_smoothness(disparity: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    disp = _as_bhw(disparity)
    disp = disp / disp.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    grad_disp_x = torch.abs(disp[:, :, 1:] - disp[:, :, :-1])
    grad_disp_y = torch.abs(disp[:, 1:, :] - disp[:, :-1, :])
    grad_img_x = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]), dim=1)
    grad_img_y = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]), dim=1)
    return (grad_disp_x * torch.exp(-grad_img_x)).mean() + (grad_disp_y * torch.exp(-grad_img_y)).mean()


def _require_geometry(batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if batch.get("focal_length") is None or batch.get("baseline") is None:
        raise ValueError("DARES reprojection loss requires focal_length and baseline in every batch.")
    return batch["focal_length"], batch["baseline"]


def _supervised_depth(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    valid = torch.isfinite(pred) & torch.isfinite(target) & (pred > 0) & (target > 0)
    if mask is not None:
        valid = valid & mask.bool()
    if not valid.any():
        return pred.new_zeros(())
    return torch.abs(torch.log(pred[valid].clamp_min(1e-6)) - torch.log(target[valid].clamp_min(1e-6))).mean()


def _supervised_disparity(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    valid = torch.isfinite(pred) & torch.isfinite(target) & (pred > 0) & (target > 0)
    if mask is not None:
        valid = valid & mask.bool()
    if not valid.any():
        return pred.new_zeros(())
    return torch.abs(torch.log(pred[valid].clamp_min(1e-6)) - torch.log(target[valid].clamp_min(1e-6))).mean()


def compute_dares_losses(batch: dict[str, Any], model_outputs: dict[str, torch.Tensor], config: dict[str, Any]) -> dict[str, torch.Tensor]:
    left = batch["left_rgb"].to(model_outputs["pred_depth_left"].device)
    right = batch["right_rgb"].to(model_outputs["pred_depth_left"].device)
    focal, baseline = _require_geometry(batch)
    focal = focal.to(left.device)
    baseline = baseline.to(left.device)
    valid_mask = batch.get("valid_mask")
    if valid_mask is not None:
        valid_mask = valid_mask.to(left.device).bool()

    reproj_cfg = config.get("reprojection", {})
    weights = config.get("weights", {})
    scales = tuple(int(scale) for scale in reproj_cfg.get("scales", [1, 2, 4]))
    alpha = float(reproj_cfg.get("alpha", 0.85))
    min_depth = float(config.get("min_depth", 1e-3))

    depth_left = _as_bhw(model_outputs["pred_depth_left"]).clamp_min(min_depth)
    disp_left = depth_to_disparity(depth_left, focal, baseline, min_depth=min_depth)
    right_recon = disparity_warp(left, disp_left, direction="left_to_right")
    mask_lr = right_recon["valid_mask"]
    if valid_mask is not None:
        mask_lr = mask_lr & valid_mask
    loss_lr = multi_scale_ssim_reprojection_loss(right_recon["warped"], right, mask_lr, scales=scales, alpha=alpha)

    zero = depth_left.new_zeros(())
    loss_rl = zero
    left_recon = None
    has_rl = False
    if bool(reproj_cfg.get("symmetric", True)):
        depth_right = model_outputs.get("pred_depth_right")
        if depth_right is not None:
            depth_right = _as_bhw(depth_right).clamp_min(min_depth)
            disp_right = depth_to_disparity(depth_right, focal, baseline, min_depth=min_depth)
            left_recon = disparity_warp(right, disp_right, direction="right_to_left")
            loss_rl = multi_scale_ssim_reprojection_loss(left_recon["warped"], left, left_recon["valid_mask"], scales=scales, alpha=alpha)
            has_rl = True

    loss_reproj = 0.5 * (loss_lr + loss_rl) if has_rl else loss_lr
    loss_smooth = edge_aware_smoothness(disp_left, left)
    loss_depth_sup = zero
    if batch.get("depth_gt") is not None:
        loss_depth_sup = _supervised_depth(depth_left, batch["depth_gt"].to(left.device), valid_mask)
    loss_disp_sup = zero
    if batch.get("disparity_gt") is not None:
        loss_disp_sup = _supervised_disparity(disp_left, batch["disparity_gt"].to(left.device), valid_mask)

    total = (
        float(weights.get("reproj", 1.0)) * loss_reproj
        + float(weights.get("smooth", 0.001)) * loss_smooth
        + float(weights.get("supervised_depth", 0.0)) * loss_depth_sup
        + float(weights.get("supervised_disparity", 0.0)) * loss_disp_sup
    )
    return {
        "loss_total": total,
        "loss_reproj_ms_ssim": loss_reproj,
        "loss_reproj_lr": loss_lr,
        "loss_reproj_rl": loss_rl,
        "loss_smooth": loss_smooth,
        "loss_depth_supervised": loss_depth_sup,
        "loss_disparity_supervised": loss_disp_sup,
        "predicted_disparity_left": disp_left,
        "right_reconstruction": right_recon["warped"],
        "left_reconstruction": left_recon["warped"] if left_recon is not None else zero.new_zeros(left.shape),
        "valid_reproj_mask": mask_lr,
    }
