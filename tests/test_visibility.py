from __future__ import annotations

import torch

from eaglevision.models.nvs.visibility import compute_projection_mask


def test_projection_mask_rejects_out_of_bounds() -> None:
    pixels = torch.tensor([[[1.0, -1.0], [1.0, 10.0]]])
    depth = torch.tensor([[1.0, 1.0]])
    valid, linear = compute_projection_mask(pixels, depth, (4, 4))
    assert valid.tolist() == [[True, False]]
    assert linear[0, 0].item() == 5
    assert linear[0, 1].item() == -1
