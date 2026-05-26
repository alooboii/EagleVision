from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from eaglevision.data.rgbd_collate import rgbd_frame_collate
from eaglevision.data.rgbd_frame_dataset import RGBDFrameDataset
from eaglevision.metrics.depth_metrics import compute_depth_metrics, depth_l1, valid_depth_mask
from eaglevision.models.depth.frozen_depth_prior import build_frozen_depth_prior
from eaglevision.models.residual_log_depth_adapter import ResidualLogDepthAdapter
from eaglevision.utils.io import ensure_dir, load_yaml


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dump frozen/adapted depth predictions for 3D-from-X evaluation.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frame-manifest", type=Path, required=True)
    parser.add_argument("--method", choices=["frozen", "adapted"], required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def _device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("device", "cuda"))
    return torch.device("cpu" if requested == "cuda" and not torch.cuda.is_available() else requested)


def _safe_frame_key(frame_id: str) -> str:
    return frame_id.replace("/", "__").replace("\\", "__")


def _resolve(root: Path, value: str | Path | None) -> str:
    if not value:
        return ""
    path = Path(value)
    return str(path if path.is_absolute() else root / path)


def _load_adapter(config: dict[str, Any], checkpoint: Path, device: torch.device) -> ResidualLogDepthAdapter:
    adapter = ResidualLogDepthAdapter(**config.get("adapter", {})).to(device)
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("adapter", payload)
    adapter.load_state_dict(state)
    adapter.eval()
    return adapter


def _save_preview(path: Path, rgb: torch.Tensor, depth: np.ndarray) -> None:
    ensure_dir(path.parent)
    valid = np.isfinite(depth) & (depth > 0)
    vis = np.zeros_like(depth, dtype=np.float32)
    if valid.any():
        vis[valid] = (depth[valid] - depth[valid].min()) / max(float(depth[valid].max() - depth[valid].min()), 1e-6)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].imshow(rgb.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy())
    axes[0].axis("off")
    axes[0].set_title("rgb")
    axes[1].imshow(vis, cmap="magma")
    axes[1].axis("off")
    axes[1].set_title("pred_depth")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    args = build_argparser().parse_args()
    config = load_yaml(args.config)
    device = _device(config)
    data_cfg = config.get("data", {})
    eval_cfg = config.get("eval", {})
    data_root = Path(data_cfg.get("root", args.frame_manifest.parent))
    dataset = RGBDFrameDataset(
        args.frame_manifest,
        root=data_root,
        image_size=tuple(data_cfg.get("image_size")) if data_cfg.get("image_size") else None,
        min_depth=float(data_cfg.get("min_depth", 1e-3)),
        max_depth=float(data_cfg.get("max_depth", 10.0)),
        depth_scale=float(data_cfg.get("depth_scale", 1000.0)),
    )
    loader = DataLoader(dataset, batch_size=int(eval_cfg.get("batch_size", 4)), shuffle=False, num_workers=int(eval_cfg.get("num_workers", 0)), collate_fn=rgbd_frame_collate)
    prior = build_frozen_depth_prior(config.get("base_model", {})).to(device).eval()
    adapter = _load_adapter(config, args.checkpoint, device) if args.method == "adapted" else None
    if args.method == "adapted" and args.checkpoint is None:
        raise ValueError("--checkpoint is required for method=adapted")

    depths_dir = ensure_dir(args.out_dir / "depths")
    previews_dir = ensure_dir(args.out_dir / "previews")
    manifest_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    preview_limit = int(eval_cfg.get("num_previews", 16))
    rows_by_id = {row.get("frame_id", f"frame_{i:06d}"): row for i, row in enumerate(dataset.rows)}
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"dump {args.method} depth"):
            images = batch["rgb"].to(device)
            base_depth = prior.predict_depth(images)
            pred = adapter(images, base_depth)["adapted_depth"] if adapter is not None else base_depth
            for i, frame_id in enumerate(batch["frame_id"]):
                scene_id = batch["scene_id"][i]
                frame_key = _safe_frame_key(frame_id)
                out_path = depths_dir / scene_id / f"{frame_key}.npy"
                ensure_dir(out_path.parent)
                depth_np = pred[i].detach().cpu().numpy().astype(np.float32)
                np.save(out_path, depth_np)
                source = rows_by_id[frame_id]
                row = {
                    "method": args.method,
                    "frame_id": frame_id,
                    "scene_id": scene_id,
                    "pred_depth": str(out_path.relative_to(args.out_dir)),
                    "rgb": _resolve(data_root, source.get("rgb")),
                    "depth_gt": _resolve(data_root, source.get("depth")),
                    "pose": _resolve(data_root, source.get("pose")),
                    "intrinsics": _resolve(data_root, source.get("intrinsics")),
                }
                manifest_rows.append(row)
                if len(manifest_rows) <= preview_limit:
                    _save_preview(previews_dir / f"{frame_key}.png", batch["rgb"][i], depth_np)
                metric_row = {"method": args.method, "frame_id": frame_id, "scene_id": scene_id}
                if batch.get("depth_gt") is not None:
                    target = batch["depth_gt"][i : i + 1]
                    pred_t = pred[i : i + 1].detach().cpu()
                    metric_row.update(compute_depth_metrics(pred_t, target, min_depth=float(data_cfg.get("min_depth", 1e-3)), max_depth=float(data_cfg.get("max_depth", 10.0))))
                    metric_row["depth_l1"] = depth_l1(pred_t, target, min_depth=float(data_cfg.get("min_depth", 1e-3)), max_depth=float(data_cfg.get("max_depth", 10.0)))
                    metric_row["num_valid_pixels"] = int(valid_depth_mask(pred_t, target, min_depth=float(data_cfg.get("min_depth", 1e-3)), max_depth=float(data_cfg.get("max_depth", 10.0))).sum().item())
                    metric_row["status"] = "ok"
                else:
                    metric_row.update({key: float("nan") for key in ("absrel", "sqrel", "rmse", "rmse_log", "silog", "delta1", "delta2", "delta3", "depth_l1")})
                    metric_row["num_valid_pixels"] = 0
                    metric_row["status"] = "missing_gt_depth"
                metrics_rows.append(metric_row)
    ensure_dir(args.out_dir)
    with (args.out_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (args.out_dir / "single_frame_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["method", "frame_id", "scene_id", "absrel", "sqrel", "rmse", "rmse_log", "silog", "delta1", "delta2", "delta3", "depth_l1", "num_valid_pixels", "status"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics_rows)
    print(f"Wrote predictions to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
