from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from eaglevision.geometry.rgbd_reprojection import load_depth_npy_or_png, load_intrinsics, load_pose, project_depth_to_target, resolve
from eaglevision.utils.io import ensure_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Known-pose geometric alignment diagnostic scaffold.")
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--pred-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def _jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    args = build_argparser().parse_args()
    pred_root = args.pred_manifest.parent
    preds = {row["frame_id"]: row for row in _jsonl(args.pred_manifest)}
    method = next(iter(preds.values()))["method"] if preds else "unknown"
    rows = []
    for pair in _jsonl(args.pair_manifest):
        out = {"method": method, "pair_id": pair["pair_id"], "scene_id": pair.get("scene_id", "")}
        try:
            src = preds[pair["source_frame_id"]]
            tgt = preds[pair["target_frame_id"]]
            depth = load_depth_npy_or_png(resolve(pred_root, src["pred_depth"]))
            src_pose = load_pose(src["pose"])
            tgt_pose = load_pose(tgt["pose"])
            K = load_intrinsics(src["intrinsics"])
            if src_pose is None or tgt_pose is None or K is None:
                raise ValueError("missing_or_invalid_pose_intrinsics")
            _, valid = project_depth_to_target(depth, src_pose, tgt_pose, K, K, depth.shape)
            out.update({"alignment_rmse": float("nan"), "alignment_median": float("nan"), "num_correspondences": int(valid.sum()), "valid_ratio": float(valid.mean()), "status": "known_pose_overlap_only"})
        except Exception as exc:
            out.update({"alignment_rmse": float("nan"), "alignment_median": float("nan"), "num_correspondences": 0, "valid_ratio": float("nan"), "status": f"error:{exc}"})
        rows.append(out)
    ensure_dir(args.out.parent)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "pair_id", "scene_id", "alignment_rmse", "alignment_median", "num_correspondences", "valid_ratio", "status"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
