from __future__ import annotations

import numpy as np


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    """Overlay a binary mask on top of an RGB image."""
    output = image.copy()
    output[mask.astype(bool)] = (0.6 * output[mask.astype(bool)] + 0.4 * np.array(color)).astype(output.dtype)
    return output
