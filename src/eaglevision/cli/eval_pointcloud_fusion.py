from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from eaglevision.geometry.pointcloud_eval import depth_to_world_points, pointcloud_metrics
from eaglevision.geometry.rgbd_reprojection import load_depth_npy_or_png, load_intrinsics, load_pose, resolve
from eaglevision.utils.io import ensure_dir


FIELDS = ["method", "scene_id", "num_frames", "num_pred_points", "num_ref_points", "chamfer_l2", "chamfer_l1", "accuracy", "completeness", "fscore_0_02", "fscore_0_05", "fscore_0_10", "status"]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate multi-frame point cloud fusion from predicted depths.")
    parser.add_argument("--frame-manifest", type=Path, required=True)
    parser.add_argument("--pred-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames-per-scene", type=int, default=20)
    parser.add_argument("--max-scenes", type=int, default=20)
    parser.add_argument("--max-points", type=int, default=200000)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    return parser


def _jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    idx = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
    return points[idx]


def main() -> int:
    args = build_argparser().parse_args()
    pred_root = args.pred_manifest.parent
    pred_rows = _jsonl(args.pred_manifest)
    method = pred_rows[0]["method"] if pred_rows else "unknown"
    by_scene: dict[str, list[dict]] = {}
    for row in pred_rows:
        by_scene.setdefault(row.get("scene_id", ""), []).append(row)
    out_rows = []
    for scene_id in sorted(by_scene)[: args.max_scenes]:
        rows = by_scene[scene_id][: args.frames_per_scene]
        out = {"method": method, "scene_id": scene_id, "num_frames": len(rows)}
        try:
            pred_points_all = []
            ref_points_all = []
            for row in rows:
                if not row.get("pose") or not row.get("intrinsics"):
                    continue
                pose = load_pose(row["pose"])
                K = load_intrinsics(row["intrinsics"])
                if pose is None or K is None:
                    continue
                pred_depth = load_depth_npy_or_png(resolve(pred_root, row["pred_depth"]))
                pred_points_all.append(depth_to_world_points(pred_depth, K, pose))
                if row.get("depth_gt"):
                    gt_depth = load_depth_npy_or_png(row["depth_gt"], depth_scale=args.depth_scale)
                    ref_points_all.append(depth_to_world_points(gt_depth, K, pose))
            pred_points = _sample_points(np.concatenate(pred_points_all, axis=0) if pred_points_all else np.zeros((0, 3), dtype=np.float32), args.max_points)
            ref_points = _sample_points(np.concatenate(ref_points_all, axis=0) if ref_points_all else np.zeros((0, 3), dtype=np.float32), args.max_points)
            out["num_pred_points"] = int(pred_points.shape[0])
            out["num_ref_points"] = int(ref_points.shape[0])
            out.update(pointcloud_metrics(pred_points, ref_points))
            out["status"] = "ok" if pred_points.size and ref_points.size else "missing_points"
        except Exception as exc:
            out.update({field: float("nan") for field in FIELDS if field not in out and field != "status"})
            out["status"] = f"error:{exc}"
        out_rows.append(out)
    ensure_dir(args.out.parent)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
