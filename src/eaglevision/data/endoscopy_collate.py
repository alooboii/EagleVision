from __future__ import annotations

from typing import Any

import torch


METADATA_KEYS = {"sample_id", "sequence_id", "frame_id"}


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
        elif all(torch.is_tensor(value) for value in values):
            batch[key] = torch.stack(values)
        elif any(value is None for value in values):
            batch[key] = None
        elif all(isinstance(value, (int, float)) for value in values):
            batch[key] = torch.tensor(values, dtype=torch.float32)
        else:
            batch[key] = values
    return batch
