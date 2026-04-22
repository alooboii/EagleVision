from __future__ import annotations

from collections import defaultdict

import torch

from eaglevision.losses.total import compute_phase1_losses
from eaglevision.metrics.consistency_metrics import depth_reprojection_consistency
from eaglevision.metrics.depth_metrics import abs_rel, depth_l1, rmse
from eaglevision.metrics.image_metrics import psnr, rgb_l1, ssim


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, dataloader, device: torch.device, loss_weights: dict[str, float]) -> dict[str, float]:
    """Run evaluation over a dataloader."""
    model.eval()
    totals: dict[str, list[float]] = defaultdict(list)
    for batch in dataloader:
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        outputs = model(batch)
        losses = compute_phase1_losses(outputs, loss_weights)
        totals["target_rgb_l1"].append(float(rgb_l1(outputs["A_t_warp"], outputs["B"], outputs["M_t"]).item()))
        totals["cycle_rgb_l1"].append(float(rgb_l1(outputs["A_s_recon"], outputs["A"], outputs["M_s"]).item()))
        totals["psnr"].append(psnr(outputs["A_s_recon"], outputs["A"], outputs["M_s"]))
        totals["ssim"].append(ssim(outputs["A_s_recon"], outputs["A"]))
        totals["reprojection_depth"].append(
            depth_reprojection_consistency(outputs["D_s_pred"], outputs["D_source_gt"], outputs["M_s"])
        )
        if outputs["D_target_gt"] is not None:
            totals["depth_l1"].append(depth_l1(outputs["D_t_pred"], outputs["D_target_gt"]))
            totals["rmse"].append(rmse(outputs["D_t_pred"], outputs["D_target_gt"]))
            totals["abs_rel"].append(abs_rel(outputs["D_t_pred"], outputs["D_target_gt"]))
        for name, value in losses.items():
            totals[name].append(float(value.item()))
    return {name: sum(values) / max(len(values), 1) for name, values in totals.items()}
