from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from eaglevision.geometry.rgbd_reprojection import depth_errors, load_depth_npy_or_png, load_intrinsics, load_pose, project_depth_to_target, resolve
from eaglevision.utils.io import ensure_dir


FIELDS = ["method", "pair_id", "scene_id", "difficulty", "translation_m", "rotation_deg", "cycle_depth_l1_pred", "cycle_depth_absrel_pred", "cycle_valid_overlap_ratio", "cycle_num_valid_pixels", "cycle_rgb_l1", "status"]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate source-target-source depth cycle consistency.")
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
    out_rows = []
    for pair in _jsonl(args.pair_manifest):
        row = {key: pair.get(key, "") for key in ("pair_id", "scene_id", "difficulty", "translation_m", "rotation_deg")}
        row["method"] = method
        try:
            src = preds[pair["source_frame_id"]]
            tgt = preds[pair["target_frame_id"]]
            src_depth = load_depth_npy_or_png(resolve(pred_root, src["pred_depth"]))
            src_pose = load_pose(src["pose"])
            tgt_pose = load_pose(tgt["pose"])
            K = load_intrinsics(src["intrinsics"])
            if src_pose is None or tgt_pose is None or K is None:
                raise ValueError("missing_or_invalid_pose_intrinsics")
            projected_tgt, valid_st = project_depth_to_target(src_depth, src_pose, tgt_pose, K, K, src_depth.shape)
            cycle_src, valid_ts = project_depth_to_target(projected_tgt, tgt_pose, src_pose, K, K, src_depth.shape)
            valid = valid_st & valid_ts
            err = depth_errors(cycle_src, src_depth, valid)
            row.update(
                {
                    "cycle_depth_l1_pred": err["depth_l1"],
                    "cycle_depth_absrel_pred": err["absrel"],
                    "cycle_valid_overlap_ratio": float(valid.mean()),
                    "cycle_num_valid_pixels": int(valid.sum()),
                    "cycle_rgb_l1": float("nan"),
                    "status": "ok",
                }
            )
        except Exception as exc:
            row.update({field: float("nan") for field in FIELDS if field not in row and field != "status"})
            row["status"] = f"error:{exc}"
        out_rows.append(row)
    ensure_dir(args.out.parent)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
