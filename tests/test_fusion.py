from __future__ import annotations

import torch

from eaglevision.models.fusion.depth_fusion import fuse_warped_and_predicted_depth


def test_fusion_uses_prediction_only_in_holes() -> None:
    warped = torch.tensor([[[1.0, 2.0]]])
    predicted = torch.tensor([[[3.0, 4.0]]])
    mask = torch.tensor([[[True, False]]])
    fused = fuse_warped_and_predicted_depth(warped, predicted, mask)
    assert torch.allclose(fused, torch.tensor([[[1.0, 4.0]]]))
