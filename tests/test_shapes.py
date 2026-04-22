from __future__ import annotations

import torch

from eaglevision.models.adapters import ResidualDepthAdapter
from eaglevision.models.fusion.depth_fusion import fuse_warped_and_predicted_depth
from eaglevision.models.nvs.geometric_warp import GeometricWarper
from eaglevision.models.rt_depthnvs import RoundTripDepthNVS


class DummyDepthModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = ResidualDepthAdapter(hidden_channels=4)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        base = torch.ones(images.shape[0], images.shape[2], images.shape[3], device=images.device)
        adapted = self.adapter(images, base)
        return {"base_depth": base, "adapted_depth": adapted}


def test_warper_and_roundtrip_shapes() -> None:
    batch = {
        "source_rgb": torch.rand(1, 3, 8, 8),
        "target_rgb": torch.rand(1, 3, 8, 8),
        "source_depth": torch.ones(1, 8, 8),
        "target_depth": torch.ones(1, 8, 8),
        "source_intrinsics": torch.eye(3).unsqueeze(0),
        "target_intrinsics": torch.eye(3).unsqueeze(0),
        "source_pose": torch.eye(4).unsqueeze(0),
        "target_pose": torch.eye(4).unsqueeze(0),
    }
    outputs = RoundTripDepthNVS(DummyDepthModel())(batch)
    assert outputs["A_t_warp"].shape == batch["source_rgb"].shape
    assert outputs["D_t_pred"].shape == batch["source_depth"].shape
    assert outputs["A_s_recon"].shape == batch["source_rgb"].shape
