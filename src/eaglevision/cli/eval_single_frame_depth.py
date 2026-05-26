from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from eaglevision.geometry.rgbd_reprojection import load_depth_npy_or_png, resolve
from eaglevision.metrics.depth_metrics import compute_depth_metrics, depth_l1, valid_depth_mask
from eaglevision.utils.io import ensure_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate single-frame predicted depth against RGB-D ground truth.")
    parser.add_argument("--pred-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-depth", type=float, default=1e-3)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    return parser


def _rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    args = build_argparser().parse_args()
    root = args.pred_manifest.parent
    out_rows = []
    for row in _rows(args.pred_manifest):
        result = {"method": row["method"], "frame_id": row["frame_id"], "scene_id": row.get("scene_id", ""), "status": "ok"}
        try:
            pred = load_depth_npy_or_png(resolve(root, row["pred_depth"]))
            if not row.get("depth_gt"):
                raise FileNotFoundError("missing depth_gt")
            target = load_depth_npy_or_png(row["depth_gt"], depth_scale=args.depth_scale)
            pred_t = torch.from_numpy(pred).unsqueeze(0)
            target_t = torch.from_numpy(target).unsqueeze(0)
            result.update(compute_depth_metrics(pred_t, target_t, min_depth=args.min_depth, max_depth=args.max_depth))
            result["depth_l1"] = depth_l1(pred_t, target_t, min_depth=args.min_depth, max_depth=args.max_depth)
            result["num_valid_pixels"] = int(valid_depth_mask(pred_t, target_t, min_depth=args.min_depth, max_depth=args.max_depth).sum().item())
        except Exception as exc:
            result.update({key: float("nan") for key in ("absrel", "sqrel", "rmse", "rmse_log", "silog", "delta1", "delta2", "delta3", "depth_l1")})
            result["num_valid_pixels"] = 0
            result["status"] = f"error:{exc}"
        out_rows.append(result)
    ensure_dir(args.out.parent)
    fields = ["method", "frame_id", "scene_id", "absrel", "sqrel", "rmse", "rmse_log", "silog", "delta1", "delta2", "delta3", "depth_l1", "num_valid_pixels", "status"]
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
