from __future__ import annotations

from typing import Any

import torch


META_KEYS = {"frame_id", "scene_id", "rgb_path", "depth_path", "pose_path", "intrinsics_path"}
OPTIONAL_TENSOR_KEYS = {"depth_gt", "valid_mask", "pose", "intrinsics"}


def rgbd_frame_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {}
    batch: dict[str, Any] = {}
    keys = sorted({key for sample in samples for key in sample})
    for key in keys:
        values = [sample.get(key) for sample in samples]
        if key in META_KEYS:
            batch[key] = ["" if value is None else str(value) for value in values]
        elif key in OPTIONAL_TENSOR_KEYS:
            present = [i for i, sample in enumerate(samples) if key in sample and sample[key] is not None]
            missing = [i for i, sample in enumerate(samples) if key not in sample or sample[key] is None]
            if present and missing:
                ids = ", ".join(str(samples[i].get("frame_id", f"index_{i}")) for i in missing)
                raise ValueError(f"Mixed optional RGB-D field '{key}' in batch; missing frame_ids: {ids}")
            batch[key] = torch.stack(values) if present else None
        elif all(torch.is_tensor(value) for value in values):
            batch[key] = torch.stack(values)
        else:
            batch[key] = values
    return batch
