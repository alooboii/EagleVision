from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

_RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class PreprocessResult:
    tensor: torch.Tensor
    original_size: tuple[int, int]


def preprocess_bgr_image(image_bgr: np.ndarray, input_size: int, device: torch.device) -> PreprocessResult:
    if image_bgr is None or image_bgr.ndim != 3:
        raise ValueError("Expected a valid BGR image array with shape (H, W, 3)")

    original_h, original_w = image_bgr.shape[:2]
    resized_w, resized_h = _compute_resize(original_w, original_h, input_size, multiple_of=14)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    resized_rgb = cv2.resize(image_rgb, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)
    normalized = (resized_rgb - _RGB_MEAN) / _RGB_STD

    chw = np.transpose(normalized, (2, 0, 1)).astype(np.float32)
    tensor = torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0).to(device)

    return PreprocessResult(tensor=tensor, original_size=(original_h, original_w))


def _compute_resize(width: int, height: int, target: int, multiple_of: int) -> tuple[int, int]:
    scale_h = target / float(height)
    scale_w = target / float(width)

    # lower_bound + keep_aspect_ratio behavior used by the official implementation
    scale = max(scale_h, scale_w)
    resized_h = _snap_to_multiple(scale * height, multiple_of, min_value=target)
    resized_w = _snap_to_multiple(scale * width, multiple_of, min_value=target)
    return resized_w, resized_h


def _snap_to_multiple(value: float, multiple_of: int, min_value: int = 0) -> int:
    snapped = int(round(value / multiple_of) * multiple_of)
    if snapped < min_value:
        snapped = int(np.ceil(value / multiple_of) * multiple_of)
    return snapped
