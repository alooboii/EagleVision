from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PairSamplingConfig:
    min_translation_m: float = 0.02
    max_translation_m: float = 0.30
    min_rotation_deg: float = 1.0
    max_rotation_deg: float = 10.0
    max_index_gap: int = 30
    frame_stride: int = 1
    max_frames_per_scene: int | None = None
    max_pairs_per_scene: int | None = None


def pose_translation_distance(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    """Compute Euclidean translation distance between two camera poses."""
    return float(np.linalg.norm(pose_a[:3, 3] - pose_b[:3, 3]))


def pose_rotation_distance_deg(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    """Compute relative rotation angle in degrees."""
    rel = pose_b[:3, :3] @ pose_a[:3, :3].T
    trace = np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)
    return float(math.degrees(math.acos(trace)))


def filter_candidate_pairs(
    poses: Iterable[np.ndarray],
    frame_ids: Iterable[int],
    config: PairSamplingConfig,
) -> list[tuple[int, int]]:
    """Return frame-id pairs that satisfy motion constraints."""
    pose_list = list(poses)
    frame_list = list(frame_ids)
    pairs: list[tuple[int, int]] = []
    for i in range(len(pose_list)):
        for j in range(i + 1, len(pose_list)):
            if frame_list[j] - frame_list[i] > config.max_index_gap:
                break
            translation = pose_translation_distance(pose_list[i], pose_list[j])
            rotation = pose_rotation_distance_deg(pose_list[i], pose_list[j])
            if not (config.min_translation_m <= translation <= config.max_translation_m):
                continue
            if not (config.min_rotation_deg <= rotation <= config.max_rotation_deg):
                continue
            pairs.append((frame_list[i], frame_list[j]))
    if config.max_pairs_per_scene is not None and len(pairs) > config.max_pairs_per_scene:
        keep = max(1, int(config.max_pairs_per_scene))
        # Uniformly downsample to keep broad motion diversity while bounding runtime.
        sampled_indices = np.linspace(0, len(pairs) - 1, num=keep, dtype=int)
        pairs = [pairs[index] for index in sampled_indices]

    return pairs
