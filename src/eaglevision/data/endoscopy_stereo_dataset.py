from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


class EndoscopyStereoDataset(Dataset):
    """JSONL manifest dataset for SCARED-style stereo endoscopy pairs."""

    def __init__(
        self,
        manifest_path: Path,
        root: Path | None = None,
        image_size: tuple[int, int] | None = None,
        min_depth: float = 1e-3,
        max_depth: float = 300.0,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = Path(root) if root is not None else self.manifest_path.parent
        self.image_size = image_size
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.rows = self._load_manifest(self.manifest_path)
        if not self.rows:
            raise ValueError(f"Manifest has no samples: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        left = self._load_rgb(row["left_rgb"])
        right = self._load_rgb(row["right_rgb"])
        original_h, original_w = left.shape[-2:]
        if right.shape[-2:] != (original_h, original_w):
            right = F.interpolate(right.unsqueeze(0), size=(original_h, original_w), mode="bilinear", align_corners=False).squeeze(0)

        sx = sy = 1.0
        if self.image_size is not None:
            target_h, target_w = self.image_size
            sy = target_h / original_h
            sx = target_w / original_w
            left = self._resize_chw(left, self.image_size, mode="bilinear")
            right = self._resize_chw(right, self.image_size, mode="bilinear")

        sample: dict[str, Any] = {
            "left_rgb": left,
            "right_rgb": right,
            "sample_id": str(row.get("sample_id", f"sample_{index:06d}")),
            "sequence_id": str(row.get("sequence_id", "")),
            "frame_id": str(row.get("frame_id", "")),
        }

        if row.get("depth"):
            depth = self._load_hw_tensor(row["depth"])
            depth = self._resize_hw(depth, self.image_size, mode="nearest") if self.image_size is not None else depth
            valid = torch.isfinite(depth) & (depth >= self.min_depth) & (depth <= self.max_depth)
            sample["depth_gt"] = depth
            sample["valid_mask"] = valid if "valid_mask" not in sample else sample["valid_mask"] & valid

        if row.get("disparity"):
            disparity = self._load_hw_tensor(row["disparity"])
            if self.image_size is not None:
                disparity = self._resize_hw(disparity, self.image_size, mode="bilinear") * sx
            sample["disparity_gt"] = disparity

        if row.get("mask"):
            mask = self._load_mask(row["mask"])
            mask = self._resize_hw(mask.float(), self.image_size, mode="nearest") > 0.5 if self.image_size is not None else mask
            sample["valid_mask"] = mask if "valid_mask" not in sample else sample["valid_mask"] & mask

        if row.get("K_left"):
            sample["K_left"] = self._scale_intrinsics(self._load_matrix(row["K_left"], (3, 3)), sx, sy)
        if row.get("K_right"):
            sample["K_right"] = self._scale_intrinsics(self._load_matrix(row["K_right"], (3, 3)), sx, sy)
        if row.get("T_left_to_right"):
            sample["T_left_to_right"] = self._load_matrix(row["T_left_to_right"], (4, 4))

        for key in ("baseline", "focal_length"):
            if row.get(key) is not None and row.get(key) != "":
                sample[key] = torch.tensor(float(row[key]), dtype=torch.float32)

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
                if "left_rgb" not in row or "right_rgb" not in row:
                    raise ValueError(f"Manifest row {line_number} must include left_rgb and right_rgb")
                rows.append(row)
        return rows

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def _load_rgb(self, value: str | Path) -> torch.Tensor:
        image = Image.open(self._resolve(value)).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def _load_hw_tensor(self, value: str | Path) -> torch.Tensor:
        path = self._resolve(value)
        if path.suffix.lower() == ".npy":
            array = np.load(path).astype(np.float32)
        else:
            array = np.asarray(Image.open(path), dtype=np.float32)
        if array.ndim == 3:
            array = array[..., 0]
        return torch.from_numpy(array.astype(np.float32))

    def _load_mask(self, value: str | Path) -> torch.Tensor:
        path = self._resolve(value)
        array = np.asarray(Image.open(path), dtype=np.float32)
        if array.ndim == 3:
            array = array[..., 0]
        return torch.from_numpy(array > 0)

    def _load_matrix(self, value: str | Path, shape: tuple[int, int]) -> torch.Tensor:
        matrix = np.loadtxt(self._resolve(value), dtype=np.float32)
        matrix = np.asarray(matrix, dtype=np.float32).reshape(shape)
        return torch.from_numpy(matrix)

    @staticmethod
    def _resize_chw(tensor: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
        kwargs = {"align_corners": False} if mode in {"bilinear", "bicubic"} else {}
        return F.interpolate(tensor.unsqueeze(0), size=size, mode=mode, **kwargs).squeeze(0)

    @staticmethod
    def _resize_hw(tensor: torch.Tensor, size: tuple[int, int] | None, mode: str) -> torch.Tensor:
        if size is None:
            return tensor
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
