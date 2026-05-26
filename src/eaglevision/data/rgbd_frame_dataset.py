from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


class RGBDFrameDataset(Dataset):
    """RGB-D frame manifest dataset for ScanNet/TUM-style inference evaluation."""

    def __init__(
        self,
        manifest_path: Path,
        root: Path | None = None,
        image_size: tuple[int, int] | None = None,
        min_depth: float = 1e-3,
        max_depth: float = 10.0,
        depth_scale: float = 1000.0,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = Path(root) if root is not None else self.manifest_path.parent
        self.image_size = image_size
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.depth_scale = float(depth_scale)
        self.rows = self._load_manifest(self.manifest_path)
        if not self.rows:
            raise ValueError(f"RGB-D frame manifest is empty: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        rgb = self._load_rgb(row["rgb"])
        original_h, original_w = rgb.shape[-2:]
        sx = sy = 1.0
        if self.image_size is not None:
            target_h, target_w = self.image_size
            sy = target_h / original_h
            sx = target_w / original_w
            rgb = self._resize_chw(rgb, self.image_size, mode="bilinear")

        sample: dict[str, Any] = {
            "frame_id": str(row.get("frame_id", f"frame_{index:06d}")),
            "scene_id": str(row.get("scene_id", "")),
            "rgb": rgb,
            "rgb_path": str(row["rgb"]),
        }
        if row.get("depth"):
            depth = self._load_depth(row["depth"])
            if self.image_size is not None:
                depth = self._resize_hw(depth, self.image_size, mode="nearest")
            valid = torch.isfinite(depth) & (depth >= self.min_depth) & (depth <= self.max_depth)
            sample["depth_gt"] = depth
            sample["valid_mask"] = valid
            sample["depth_path"] = str(row["depth"])
        if row.get("pose"):
            sample["pose"] = self._load_matrix(row["pose"], (4, 4))
            sample["pose_path"] = str(row["pose"])
        if row.get("intrinsics"):
            intrinsics = self._load_matrix(row["intrinsics"], (3, 3))
            sample["intrinsics"] = self._scale_intrinsics(intrinsics, sx, sy)
            sample["intrinsics_path"] = str(row["intrinsics"])
        return sample

    @staticmethod
    def _load_manifest(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                if "rgb" not in row:
                    raise ValueError(f"Manifest row {line_number} must include rgb")
                rows.append(row)
        return rows

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def _load_rgb(self, value: str | Path) -> torch.Tensor:
        image = Image.open(self._resolve(value)).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def _load_depth(self, value: str | Path) -> torch.Tensor:
        path = self._resolve(value)
        if path.suffix.lower() == ".npy":
            array = np.load(path).astype(np.float32)
        else:
            array = np.asarray(Image.open(path), dtype=np.float32) / self.depth_scale
        if array.ndim == 3:
            array = array[..., 0]
        return torch.from_numpy(array.astype(np.float32))

    def _load_matrix(self, value: str | Path, shape: tuple[int, int]) -> torch.Tensor:
        matrix = np.loadtxt(self._resolve(value), dtype=np.float32).reshape(shape)
        return torch.from_numpy(matrix)

    @staticmethod
    def _resize_chw(tensor: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
        kwargs = {"align_corners": False} if mode in {"bilinear", "bicubic"} else {}
        return F.interpolate(tensor.unsqueeze(0), size=size, mode=mode, **kwargs).squeeze(0)

    @staticmethod
    def _resize_hw(tensor: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
        kwargs = {"align_corners": False} if mode in {"bilinear", "bicubic"} else {}
        return F.interpolate(tensor.unsqueeze(0).unsqueeze(0).float(), size=size, mode=mode, **kwargs).squeeze(0).squeeze(0)

    @staticmethod
    def _scale_intrinsics(matrix: torch.Tensor, sx: float, sy: float) -> torch.Tensor:
        scaled = matrix.clone()
        scaled[0, 0] *= sx
        scaled[0, 2] *= sx
        scaled[1, 1] *= sy
        scaled[1, 2] *= sy
        return scaled
