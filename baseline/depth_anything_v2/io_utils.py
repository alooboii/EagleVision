from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImageCollection:
    root: Path
    images: list[Path]


def collect_images(input_path: Path) -> ImageCollection:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image file extension: {input_path.suffix}")
        return ImageCollection(root=input_path.parent, images=[input_path])

    images = sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No supported image files found under: {input_path}")

    return ImageCollection(root=input_path, images=images)


def relative_output_stem(image_path: Path, input_path: Path) -> Path:
    if input_path.is_file():
        return Path(image_path.stem)

    rel = image_path.relative_to(input_path)
    return rel.with_suffix("")


def to_depth_preview(depth: np.ndarray) -> np.ndarray:
    safe_depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_depth = float(np.min(safe_depth))
    max_depth = float(np.max(safe_depth))

    if max_depth - min_depth < 1e-12:
        normalized = np.zeros_like(safe_depth, dtype=np.uint8)
    else:
        normalized = np.clip((safe_depth - min_depth) / (max_depth - min_depth), 0.0, 1.0)
        normalized = (normalized * 255.0).astype(np.uint8)

    return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)


def save_outputs(output_base: Path, depth: np.ndarray) -> tuple[Path, Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)

    npy_path = output_base.with_name(output_base.name + "_depth.npy")
    png_path = output_base.with_name(output_base.name + "_depth.png")

    np.save(npy_path, depth.astype(np.float32))
    preview = to_depth_preview(depth)
    cv2.imwrite(str(png_path), preview)

    return npy_path, png_path
