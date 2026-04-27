from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from eaglevision.data.pair_sampler import PairSamplingConfig, filter_candidate_pairs
from eaglevision.data.transforms import depth_to_tensor, image_to_tensor, resize_sample


@dataclass(frozen=True)
class FrameRecord:
    frame_id: int
    frame_key: str
    rgb_path: Path
    depth_path: Path
    pose_path: Path


class ScanNetPairDataset(Dataset[dict[str, Any]]):
    """ScanNet-style paired-view dataset with reproducible scene-level pairing."""

    def __init__(
        self,
        root: Path,
        scenes: list[str],
        image_size: tuple[int, int],
        intrinsics: np.ndarray,
        pair_config: PairSamplingConfig,
    ) -> None:
        self.root = root
        self.scenes = scenes
        self.image_size = image_size
        self.base_intrinsics = intrinsics.astype(np.float32)
        self.pair_config = pair_config
        self.samples = self._build_index()

    def _build_index(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for scene_id in self.scenes:
            scene_dir = self.root / scene_id
            records = self._scene_records(scene_dir)
            poses = [self._load_pose(record.pose_path) for record in records]
            frame_ids = [record.frame_id for record in records]
            valid_pairs = filter_candidate_pairs(poses, frame_ids, self.pair_config)
            record_map = {record.frame_id: record for record in records}
            pose_map = {record.frame_id: pose for record, pose in zip(records, poses, strict=True)}
            for src_id, tgt_id in valid_pairs:
                samples.append(
                    {
                        "scene_id": scene_id,
                        "source_record": record_map[src_id],
                        "target_record": record_map[tgt_id],
                        "source_pose": pose_map[src_id],
                        "target_pose": pose_map[tgt_id],
                    }
                )
        return samples

    def _scene_records(self, scene_dir: Path) -> list[FrameRecord]:
        color_dir = scene_dir / "color"
        depth_dir = scene_dir / "depth"
        pose_dir = scene_dir / "pose"
        color_paths = list(color_dir.glob("*.jpg")) + list(color_dir.glob("*.jpeg")) + list(color_dir.glob("*.png"))
        depth_paths = list(depth_dir.glob("*.png")) + list(depth_dir.glob("*.npy"))
        pose_paths = list(pose_dir.glob("*.txt"))

        color_map = {path.stem: path for path in color_paths}
        depth_map = {path.stem: path for path in depth_paths}
        pose_map = {path.stem: path for path in pose_paths}

        common_stems = sorted(
            set(color_map).intersection(depth_map).intersection(pose_map),
            key=self._stem_sort_key,
        )
        frame_stride = max(1, int(self.pair_config.frame_stride))
        common_stems = common_stems[::frame_stride]
        if self.pair_config.max_frames_per_scene is not None:
            common_stems = common_stems[: int(self.pair_config.max_frames_per_scene)]

        records = [
            FrameRecord(
                frame_id=frame_idx,
                frame_key=frame_key,
                rgb_path=color_map[frame_key],
                depth_path=depth_map[frame_key],
                pose_path=pose_map[frame_key],
            )
            for frame_idx, frame_key in enumerate(common_stems)
        ]
        return records

    @staticmethod
    def _stem_sort_key(stem: str) -> tuple[int, int | str]:
        try:
            return (0, int(stem))
        except ValueError:
            return (1, stem)

    @staticmethod
    def _load_pose(path: Path) -> np.ndarray:
        return np.loadtxt(path, dtype=np.float32).reshape(4, 4)

    @staticmethod
    def _load_rgb(path: Path) -> np.ndarray:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _load_depth(path: Path) -> np.ndarray:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(path)
        depth = depth.astype(np.float32)
        if depth.max() > 1000:
            depth = depth / 1000.0
        return depth

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        source_rgb = self._load_rgb(sample["source_record"].rgb_path)
        target_rgb = self._load_rgb(sample["target_record"].rgb_path)
        source_depth = self._load_depth(sample["source_record"].depth_path)
        target_depth = self._load_depth(sample["target_record"].depth_path)

        src = resize_sample(source_rgb, source_depth, self.base_intrinsics, self.image_size)
        tgt = resize_sample(target_rgb, target_depth, self.base_intrinsics, self.image_size)

        return {
            "source_rgb": image_to_tensor(src.image),
            "target_rgb": image_to_tensor(tgt.image),
            "source_depth": depth_to_tensor(src.depth),
            "target_depth": depth_to_tensor(tgt.depth),
            "source_intrinsics": torch.from_numpy(src.intrinsics).float(),
            "target_intrinsics": torch.from_numpy(tgt.intrinsics).float(),
            "source_pose": torch.from_numpy(sample["source_pose"]).float(),
            "target_pose": torch.from_numpy(sample["target_pose"]).float(),
            "scene_id": sample["scene_id"],
            "source_frame_id": sample["source_record"].frame_id,
            "target_frame_id": sample["target_record"].frame_id,
            "source_frame_key": sample["source_record"].frame_key,
            "target_frame_key": sample["target_record"].frame_key,
        }
