from __future__ import annotations

import torch
from torch import nn

from eaglevision.models.depth.depth_anything_wrapper import DepthAnythingWithAdapter
from eaglevision.models.fusion.depth_fusion import fuse_warped_and_predicted_depth
from eaglevision.models.nvs.geometric_warp import GeometricWarper


class RoundTripDepthNVS(nn.Module):
    """Phase 1 round-trip geometry system centered on depth adaptation."""

    def __init__(self, depth_model: DepthAnythingWithAdapter) -> None:
        super().__init__()
        self.depth_model = depth_model
        self.warper = GeometricWarper()

    @staticmethod
    def relative_pose(source_pose: torch.Tensor, target_pose: torch.Tensor) -> torch.Tensor:
        return target_pose @ torch.inverse(source_pose)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        a = batch["source_rgb"]
        b = batch["target_rgb"]
        d_source_gt = batch["source_depth"]
        k_s = batch["source_intrinsics"]
        k_t = batch["target_intrinsics"]
        p_s = batch["source_pose"]
        p_t = batch["target_pose"]

        t_s2t = self.relative_pose(p_s, p_t)
        t_t2s = self.relative_pose(p_t, p_s)

        forward = self.warper(a, d_source_gt, k_s, k_t, t_s2t)
        a_t_warp = forward["warped_rgb"]
        d_t_warp = forward["warped_depth"]
        m_t = forward["valid_mask"]
        h_t = forward["hole_mask"]

        depth_target = self.depth_model(a_t_warp)
        d_t_pred = depth_target["adapted_depth"]
        d_t_pred_base = depth_target["base_depth"]
        d_t_fused = fuse_warped_and_predicted_depth(d_t_warp, d_t_pred, m_t)

        backward = self.warper(a_t_warp, d_t_fused, k_t, k_s, t_t2s)
        a_s_recon = backward["warped_rgb"]
        d_s_warp = backward["warped_depth"]
        m_s = backward["valid_mask"]
        h_s = backward["hole_mask"]

        depth_source = self.depth_model(a_s_recon)
        d_s_pred = depth_source["adapted_depth"]

        return {
            "A": a,
            "B": b,
            "D_source_gt": d_source_gt,
            "D_target_gt": batch.get("target_depth"),
            "A_t_warp": a_t_warp,
            "D_t_warp": d_t_warp,
            "M_t": m_t,
            "H_t": h_t,
            "D_t_pred": d_t_pred,
            "D_t_pred_base": d_t_pred_base,
            "D_t_fused": d_t_fused,
            "A_s_recon": a_s_recon,
            "D_s_warp": d_s_warp,
            "D_s_pred": d_s_pred,
            "M_s": m_s,
            "H_s": h_s,
            "T_s2t": t_s2t,
            "T_t2s": t_t2s,
        }
