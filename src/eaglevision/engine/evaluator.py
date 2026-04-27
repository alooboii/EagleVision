from __future__ import annotations

from collections import defaultdict

import torch
from tqdm import tqdm

from eaglevision.losses.total import compute_phase1_losses
from eaglevision.metrics.consistency_metrics import depth_reprojection_consistency
from eaglevision.metrics.depth_metrics import abs_rel, depth_l1, rmse
from eaglevision.metrics.image_metrics import psnr, rgb_l1, ssim


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    loss_weights: dict[str, float],
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run evaluation over a dataloader."""
    model.eval()
    totals: dict[str, list[float]] = defaultdict(list)
    total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
    if max_batches is not None and total_batches is not None:
        progress_total = min(total_batches, max_batches)
    else:
        progress_total = total_batches
    print(
        f"Starting evaluation on device={device} with "
        f"{progress_total if progress_total is not None else 'unknown'} batches"
    )
    progress = tqdm(dataloader, total=progress_total, desc="eval", leave=True)
    for batch_index, batch in enumerate(progress, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        outputs = model(batch)
        losses = compute_phase1_losses(outputs, loss_weights)
        target_rgb_l1 = float(rgb_l1(outputs["A_t_warp"], outputs["B"], outputs["M_t"]).item())
        cycle_rgb_l1 = float(rgb_l1(outputs["A_s_recon"], outputs["A"], outputs["M_s"]).item())
        psnr_value = psnr(outputs["A_s_recon"], outputs["A"], outputs["M_s"])
        ssim_value = ssim(outputs["A_s_recon"], outputs["A"])
        reprojection_depth = depth_reprojection_consistency(outputs["D_s_pred"], outputs["D_source_gt"], outputs["M_s"])

        totals["target_rgb_l1"].append(target_rgb_l1)
        totals["cycle_rgb_l1"].append(cycle_rgb_l1)
        totals["psnr"].append(psnr_value)
        totals["ssim"].append(ssim_value)
        totals["reprojection_depth"].append(reprojection_depth)
        if outputs["D_target_gt"] is not None:
            totals["depth_l1"].append(depth_l1(outputs["D_t_pred"], outputs["D_target_gt"]))
            totals["rmse"].append(rmse(outputs["D_t_pred"], outputs["D_target_gt"]))
            totals["abs_rel"].append(abs_rel(outputs["D_t_pred"], outputs["D_target_gt"]))
        for name, value in losses.items():
            totals[name].append(float(value.item()))
        progress.set_postfix(
            target_rgb_l1=f"{target_rgb_l1:.4f}",
            cycle_rgb_l1=f"{cycle_rgb_l1:.4f}",
            reproj=f"{reprojection_depth:.4f}",
        )
        if batch_index == 1 or batch_index % 10 == 0:
            print(
                f"[eval] batch {batch_index}/{total_batches or '?'} "
                f"target_rgb_l1={target_rgb_l1:.4f} cycle_rgb_l1={cycle_rgb_l1:.4f} "
                f"psnr={psnr_value:.2f} ssim={ssim_value:.4f} reprojection_depth={reprojection_depth:.4f}"
            )
    return {name: sum(values) / max(len(values), 1) for name, values in totals.items()}
