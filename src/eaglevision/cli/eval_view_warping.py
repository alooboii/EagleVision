from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from eaglevision.geometry.rgbd_reprojection import load_depth_npy_or_png, load_intrinsics, load_pose, load_rgb, resolve, warp_rgb_to_target
from eaglevision.utils.io import ensure_dir


FIELDS = ["method", "pair_id", "scene_id", "difficulty", "translation_m", "rotation_deg", "view_rgb_l1", "view_rgb_psnr", "view_rgb_ssim", "valid_warp_ratio", "hole_ratio", "num_valid_pixels", "status"]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate geometric novel-view RGB warping diagnostics.")
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
            src_rgb = load_rgb(src["rgb"])
            tgt_rgb = load_rgb(tgt["rgb"])
            src_depth = load_depth_npy_or_png(resolve(pred_root, src["pred_depth"]))
            src_pose = load_pose(src["pose"])
            tgt_pose = load_pose(tgt["pose"])
            K = load_intrinsics(src["intrinsics"])
            if src_pose is None or tgt_pose is None or K is None:
                raise ValueError("missing_or_invalid_pose_intrinsics")
            warped, valid = warp_rgb_to_target(src_rgb, src_depth, src_pose, tgt_pose, K, K, tgt_rgb.shape[:2])
            if valid.any():
                diff = np.abs(warped[valid] - tgt_rgb[valid])
                l1 = float(np.mean(diff))
                mse = float(np.mean((warped[valid] - tgt_rgb[valid]) ** 2))
                psnr = float(-10.0 * math.log10(max(mse, 1e-12)))
            else:
                l1 = psnr = float("nan")
            row.update({"view_rgb_l1": l1, "view_rgb_psnr": psnr, "view_rgb_ssim": float("nan"), "valid_warp_ratio": float(valid.mean()), "hole_ratio": float(1.0 - valid.mean()), "num_valid_pixels": int(valid.sum()), "status": "ok"})
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
