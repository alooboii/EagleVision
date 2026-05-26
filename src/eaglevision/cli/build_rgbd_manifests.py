from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build RGB-D frame and pair manifests for 3D-from-X evaluation.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset-format", choices=["scannet"], default="scannet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames-per-scene", type=int, default=200)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--min-translation-m", type=float, default=0.02)
    parser.add_argument("--max-translation-m", type=float, default=0.30)
    parser.add_argument("--min-rotation-deg", type=float, default=0.5)
    parser.add_argument("--max-rotation-deg", type=float, default=10.0)
    parser.add_argument("--max-pairs-per-scene", type=int, default=100)
    return parser


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _load_pose(path: Path) -> np.ndarray | None:
    try:
        pose = np.loadtxt(path, dtype=np.float64).reshape(4, 4)
    except Exception:
        return None
    return pose if np.isfinite(pose).all() else None


def _rotation_deg(relative: np.ndarray) -> float:
    trace = float(np.trace(relative[:3, :3]))
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    return math.degrees(math.acos(cos_theta))


def _difficulty(translation: float, rotation: float) -> str:
    if translation <= 0.10 and rotation <= 3.0:
        return "easy"
    if translation <= 0.20 and rotation <= 6.0:
        return "medium"
    return "hard"


def _frame_stem(path: Path) -> str:
    return path.stem


def _discover_scannet_scene(root: Path, scene: Path, max_frames: int, frame_stride: int) -> list[dict[str, Any]]:
    color_dir = scene / "color"
    depth_dir = scene / "depth"
    pose_dir = scene / "pose"
    intrinsic_dir = scene / "intrinsic"
    if not color_dir.exists():
        return []
    intrinsic = intrinsic_dir / "intrinsic_depth.txt"
    if not intrinsic.exists():
        intrinsic = intrinsic_dir / "intrinsic_color.txt"
    rows = []
    rgb_files = sorted([path for path in color_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    for rgb_path in rgb_files[:: max(frame_stride, 1)]:
        stem = _frame_stem(rgb_path)
        depth_path = depth_dir / f"{stem}.png"
        pose_path = pose_dir / f"{stem}.txt"
        if not depth_path.exists() or not pose_path.exists():
            continue
        row = {
            "frame_id": f"{scene.name}/{stem}",
            "scene_id": scene.name,
            "rgb": _rel(rgb_path, root),
            "depth": _rel(depth_path, root),
            "pose": _rel(pose_path, root),
            "intrinsics": _rel(intrinsic, root) if intrinsic.exists() else "",
        }
        rows.append(row)
        if len(rows) >= max_frames:
            break
    return rows


def _build_pairs(root: Path, frames: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for frame in frames:
        by_scene.setdefault(frame["scene_id"], []).append(frame)
    pairs: list[dict[str, Any]] = []
    for scene_id, scene_frames in sorted(by_scene.items()):
        scene_pairs: list[dict[str, Any]] = []
        poses = {frame["frame_id"]: _load_pose(root / frame["pose"]) for frame in scene_frames}
        for i, src in enumerate(scene_frames):
            src_pose = poses[src["frame_id"]]
            if src_pose is None:
                continue
            for tgt in scene_frames[i + 1 :]:
                tgt_pose = poses[tgt["frame_id"]]
                if tgt_pose is None:
                    continue
                rel = np.linalg.inv(tgt_pose) @ src_pose
                trans = float(np.linalg.norm(rel[:3, 3]))
                rot = _rotation_deg(rel)
                if not (args.min_translation_m <= trans <= args.max_translation_m):
                    continue
                if not (args.min_rotation_deg <= rot <= args.max_rotation_deg):
                    continue
                src_key = Path(src["frame_id"]).name
                tgt_key = Path(tgt["frame_id"]).name
                scene_pairs.append(
                    {
                        "pair_id": f"{scene_id}__{src_key}__{tgt_key}",
                        "scene_id": scene_id,
                        "source_frame_id": src["frame_id"],
                        "target_frame_id": tgt["frame_id"],
                        "source_rgb": src["rgb"],
                        "target_rgb": tgt["rgb"],
                        "source_depth": src.get("depth", ""),
                        "target_depth": tgt.get("depth", ""),
                        "source_pose": src.get("pose", ""),
                        "target_pose": tgt.get("pose", ""),
                        "intrinsics": src.get("intrinsics", ""),
                        "translation_m": trans,
                        "rotation_deg": rot,
                        "difficulty": _difficulty(trans, rot),
                    }
                )
        pairs.extend(scene_pairs[: args.max_pairs_per_scene])
    return pairs


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    args = build_argparser().parse_args()
    root = args.root
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scenes = sorted([path for path in root.iterdir() if path.is_dir() and path.name.startswith("scene")])
    rng = random.Random(args.seed)
    rng.shuffle(scenes)
    val_count = max(1, int(round(0.2 * len(scenes)))) if scenes else 0
    val_scenes = set(scene.name for scene in scenes[:val_count])
    all_frames: list[dict[str, Any]] = []
    for scene in sorted(scenes):
        all_frames.extend(_discover_scannet_scene(root, scene, args.max_frames_per_scene, args.frame_stride))
    train_frames = [row for row in all_frames if row["scene_id"] not in val_scenes]
    val_frames = [row for row in all_frames if row["scene_id"] in val_scenes]
    pairs_val = _build_pairs(root, val_frames, args)
    _write_jsonl(args.out_dir / "frames_train.jsonl", train_frames)
    _write_jsonl(args.out_dir / "frames_val.jsonl", val_frames)
    _write_jsonl(args.out_dir / "pairs_val.jsonl", pairs_val)
    summary = {
        "root": str(root),
        "num_scenes": len(scenes),
        "train_frames": len(train_frames),
        "val_frames": len(val_frames),
        "val_pairs": len(pairs_val),
        "val_scenes": sorted(val_scenes),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "summary.md").write_text(
        "\n".join(
            [
                "# RGB-D Manifest Summary",
                "",
                f"- Root: `{root}`",
                f"- Scenes: {len(scenes)}",
                f"- Train frames: {len(train_frames)}",
                f"- Val frames: {len(val_frames)}",
                f"- Val pairs: {len(pairs_val)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote RGB-D manifests to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
