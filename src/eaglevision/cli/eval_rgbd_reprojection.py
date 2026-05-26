from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from eaglevision.geometry.rgbd_reprojection import depth_errors, load_depth_npy_or_png, load_intrinsics, load_pose, project_depth_to_target, resolve
from eaglevision.utils.io import ensure_dir


FIELDS = [
    "method", "pair_id", "scene_id", "difficulty", "translation_m", "rotation_deg",
    "source_absrel", "target_absrel", "source_rmse", "target_rmse", "source_delta1", "target_delta1",
    "src_to_tgt_depth_l1_gt", "src_to_tgt_absrel_gt", "src_to_tgt_depth_l1_pred", "src_to_tgt_absrel_pred",
    "tgt_to_src_depth_l1_gt", "tgt_to_src_absrel_gt", "tgt_to_src_depth_l1_pred", "tgt_to_src_absrel_pred",
    "valid_overlap_ratio_src_to_tgt", "valid_overlap_ratio_tgt_to_src", "num_valid_pixels_src_to_tgt", "num_valid_pixels_tgt_to_src", "status",
]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate pairwise RGB-D reprojection consistency.")
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--pred-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    return parser


def _jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_pred(row: dict, root: Path) -> np.ndarray:
    return load_depth_npy_or_png(resolve(root, row["pred_depth"]))


def _single(pred: np.ndarray, gt_path: str, depth_scale: float) -> dict[str, float]:
    if not gt_path:
        return {"absrel": float("nan"), "rmse": float("nan"), "delta1": float("nan")}
    return depth_errors(pred, load_depth_npy_or_png(gt_path, depth_scale=depth_scale))


def _project_errors(src_depth: np.ndarray, tgt_depth_pred: np.ndarray, tgt_depth_gt_path: str, src_pose, tgt_pose, K, depth_scale: float) -> dict[str, float]:
    projected, valid = project_depth_to_target(src_depth, src_pose, tgt_pose, K, K, tgt_depth_pred.shape)
    pred_err = depth_errors(projected, tgt_depth_pred, valid)
    gt_err = depth_errors(projected, load_depth_npy_or_png(tgt_depth_gt_path, depth_scale=depth_scale), valid) if tgt_depth_gt_path else {"depth_l1": float("nan"), "absrel": float("nan"), "num_valid_pixels": 0}
    return {
        "depth_l1_gt": gt_err["depth_l1"],
        "absrel_gt": gt_err["absrel"],
        "depth_l1_pred": pred_err["depth_l1"],
        "absrel_pred": pred_err["absrel"],
        "valid_overlap_ratio": float(valid.mean()),
        "num_valid_pixels": int(valid.sum()),
    }


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
            src_depth = _load_pred(src, pred_root)
            tgt_depth = _load_pred(tgt, pred_root)
            src_pose = load_pose(src["pose"])
            tgt_pose = load_pose(tgt["pose"])
            K = load_intrinsics(src["intrinsics"])
            if src_pose is None or tgt_pose is None or K is None:
                raise ValueError("missing_or_invalid_pose_intrinsics")
            src_single = _single(src_depth, src.get("depth_gt", ""), args.depth_scale)
            tgt_single = _single(tgt_depth, tgt.get("depth_gt", ""), args.depth_scale)
            st = _project_errors(src_depth, tgt_depth, tgt.get("depth_gt", ""), src_pose, tgt_pose, K, args.depth_scale)
            ts = _project_errors(tgt_depth, src_depth, src.get("depth_gt", ""), tgt_pose, src_pose, K, args.depth_scale)
            row.update(
                {
                    "source_absrel": src_single["absrel"],
                    "target_absrel": tgt_single["absrel"],
                    "source_rmse": src_single["rmse"],
                    "target_rmse": tgt_single["rmse"],
                    "source_delta1": src_single["delta1"],
                    "target_delta1": tgt_single["delta1"],
                    "src_to_tgt_depth_l1_gt": st["depth_l1_gt"],
                    "src_to_tgt_absrel_gt": st["absrel_gt"],
                    "src_to_tgt_depth_l1_pred": st["depth_l1_pred"],
                    "src_to_tgt_absrel_pred": st["absrel_pred"],
                    "tgt_to_src_depth_l1_gt": ts["depth_l1_gt"],
                    "tgt_to_src_absrel_gt": ts["absrel_gt"],
                    "tgt_to_src_depth_l1_pred": ts["depth_l1_pred"],
                    "tgt_to_src_absrel_pred": ts["absrel_pred"],
                    "valid_overlap_ratio_src_to_tgt": st["valid_overlap_ratio"],
                    "valid_overlap_ratio_tgt_to_src": ts["valid_overlap_ratio"],
                    "num_valid_pixels_src_to_tgt": st["num_valid_pixels"],
                    "num_valid_pixels_tgt_to_src": ts["num_valid_pixels"],
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
