from __future__ import annotations

import torch

from eaglevision.losses.depth import masked_depth_l1
from eaglevision.losses.photometric import masked_l1


def compute_phase1_losses(outputs: dict[str, torch.Tensor], weights: dict[str, float]) -> dict[str, torch.Tensor]:
    """Compute the default Phase 1 training losses."""
    target_rgb = masked_l1(outputs["A_t_warp"], outputs["B"], outputs["M_t"])
    cycle_rgb = masked_l1(outputs["A_s_recon"], outputs["A"], outputs["M_s"])
    cycle_depth = masked_depth_l1(outputs["D_s_pred"], outputs["D_source_gt"], outputs["M_s"])

    if outputs["D_target_gt"] is not None:
        target_depth = masked_depth_l1(outputs["D_t_pred"], outputs["D_target_gt"])
    else:
        target_depth = torch.zeros_like(target_rgb)

    total = (
        weights["target_rgb"] * target_rgb
        + weights["cycle_rgb"] * cycle_rgb
        + weights["cycle_depth"] * cycle_depth
        + weights["target_depth"] * target_depth
    )
    return {
        "loss_total": total,
        "loss_target_rgb": target_rgb,
        "loss_cycle_rgb": cycle_rgb,
        "loss_cycle_depth": cycle_depth,
        "loss_target_depth": target_depth,
    }
