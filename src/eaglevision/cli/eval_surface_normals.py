from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from eaglevision.geometry.rgbd_reprojection import backproject_depth, load_depth_npy_or_png, load_intrinsics, resolve
from eaglevision.utils.io import ensure_dir


FIELDS = ["method", "frame_id", "scene_id", "normal_mae_deg", "normal_median_deg", "normal_11_25", "normal_22_5", "normal_30", "num_valid_pixels", "status"]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate surface normal consistency from predicted depth.")
    parser.add_argument("--pred-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    return parser


def _jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normals_from_depth(depth: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    points_flat, valid = backproject_depth(depth, K)
    points = np.full((h * w, 3), np.nan, dtype=np.float32)
    points[valid.reshape(-1)] = points_flat
    points = points.reshape(h, w, 3)
    dx = points[:, 2:, :] - points[:, :-2, :]
    dy = points[2:, :, :] - points[:-2, :, :]
    dx = dx[1:-1]
    dy = dy[:, 1:-1]
    normals = np.cross(dx, dy)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(norm, 1e-8)
    mask = np.isfinite(normals).all(axis=-1) & (norm[..., 0] > 1e-8)
    return normals, mask


def normal_metrics(pred_depth: np.ndarray, gt_depth: np.ndarray, K: np.ndarray) -> dict[str, float]:
    pred_n, pred_mask = normals_from_depth(pred_depth, K)
    gt_n, gt_mask = normals_from_depth(gt_depth, K)
    mask = pred_mask & gt_mask
    if not mask.any():
        return {"normal_mae_deg": float("nan"), "normal_median_deg": float("nan"), "normal_11_25": float("nan"), "normal_22_5": float("nan"), "normal_30": float("nan"), "num_valid_pixels": 0}
    dot = np.abs(np.sum(pred_n[mask] * gt_n[mask], axis=-1)).clip(0, 1)
    deg = np.degrees(np.arccos(dot))
    return {
        "normal_mae_deg": float(np.mean(deg)),
        "normal_median_deg": float(np.median(deg)),
        "normal_11_25": float(np.mean(deg < 11.25)),
        "normal_22_5": float(np.mean(deg < 22.5)),
        "normal_30": float(np.mean(deg < 30.0)),
        "num_valid_pixels": int(mask.sum()),
    }


def main() -> int:
    args = build_argparser().parse_args()
    root = args.pred_manifest.parent
    rows = []
    for row in _jsonl(args.pred_manifest):
        out = {"method": row["method"], "frame_id": row["frame_id"], "scene_id": row.get("scene_id", "")}
        try:
            if not row.get("depth_gt") or not row.get("intrinsics"):
                raise ValueError("missing_gt_depth_or_intrinsics")
            pred = load_depth_npy_or_png(resolve(root, row["pred_depth"]))
            gt = load_depth_npy_or_png(row["depth_gt"], depth_scale=args.depth_scale)
            K = load_intrinsics(row["intrinsics"])
            if K is None:
                raise ValueError("invalid_intrinsics")
            out.update(normal_metrics(pred, gt, K))
            out["status"] = "ok"
        except Exception as exc:
            out.update({field: float("nan") for field in FIELDS if field not in out and field not in {"status", "num_valid_pixels"}})
            out["num_valid_pixels"] = 0
            out["status"] = f"error:{exc}"
        rows.append(out)
    ensure_dir(args.out.parent)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
