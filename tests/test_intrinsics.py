from __future__ import annotations

import torch

from eaglevision.utils.intrinsics import scale_intrinsics


def test_scale_intrinsics_updates_focal_and_principal_point() -> None:
    intrinsics = torch.tensor([[100.0, 0.0, 50.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]])
    scaled = scale_intrinsics(intrinsics, (100, 200), (50, 100))
    assert torch.allclose(scaled[0], torch.tensor([50.0, 0.0, 25.0]))
    assert torch.allclose(scaled[1], torch.tensor([0.0, 60.0, 30.0]))
