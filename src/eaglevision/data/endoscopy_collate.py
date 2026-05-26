from __future__ import annotations

from typing import Any

import torch


METADATA_KEYS = {"sample_id", "sequence_id", "frame_id"}
OPTIONAL_TENSOR_KEYS = {
    "depth_gt",
    "disparity_gt",
    "valid_mask",
    "K_left",
    "K_right",
    "T_left_to_right",
    "baseline",
    "focal_length",
}


def _sample_ids(samples: list[dict[str, Any]], indices: list[int]) -> str:
    names = [str(samples[index].get("sample_id", f"index_{index}")) for index in indices]
    return ", ".join(names)


def endoscopy_stereo_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate stereo manifest samples with optional fields."""
    if not samples:
        return {}

    batch: dict[str, Any] = {}
    keys = sorted({key for sample in samples for key in sample})
    for key in keys:
        values = [sample.get(key) for sample in samples]
        if key in METADATA_KEYS:
            batch[key] = ["" if value is None else str(value) for value in values]
        elif key in OPTIONAL_TENSOR_KEYS:
            present = [index for index, sample in enumerate(samples) if key in sample and sample[key] is not None]
            missing = [index for index, sample in enumerate(samples) if key not in sample or sample[key] is None]
            if present and missing:
                raise ValueError(
                    f"Mixed optional field '{key}' in endoscopy batch is not supported. "
                    f"Missing for sample_ids: {_sample_ids(samples, missing)}"
                )
            if not present:
                batch[key] = None
            elif all(torch.is_tensor(value) for value in values):
                batch[key] = torch.stack(values)
            else:
                batch[key] = torch.tensor(values, dtype=torch.float32)
        elif all(torch.is_tensor(value) for value in values):
            batch[key] = torch.stack(values)
        elif any(value is None for value in values):
            missing = [index for index, value in enumerate(values) if value is None]
            raise ValueError(
                f"Mixed field '{key}' in endoscopy batch is not supported. "
                f"Missing for sample_ids: {_sample_ids(samples, missing)}"
            )
        elif all(isinstance(value, (int, float)) for value in values):
            batch[key] = torch.tensor(values, dtype=torch.float32)
        else:
            batch[key] = values
    return batch
