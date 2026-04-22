from __future__ import annotations

from typing import Any

import torch


def scannet_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate fixed-shape tensors and keep metadata as lists."""
    tensor_keys = {
        "source_rgb",
        "target_rgb",
        "source_depth",
        "target_depth",
        "source_intrinsics",
        "target_intrinsics",
        "source_pose",
        "target_pose",
    }
    output: dict[str, Any] = {}
    for key in batch[0]:
        if key in tensor_keys:
            output[key] = torch.stack([item[key] for item in batch], dim=0)
        else:
            output[key] = [item[key] for item in batch]
    return output
