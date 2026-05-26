from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from eaglevision.dares.depth_anything_dares import DARESDepthAnything
from eaglevision.dares.metrics import compute_depth_metrics, disparity_metrics, reprojection_metrics
from eaglevision.dares.vector_lora import load_vector_lora_state_dict
from eaglevision.data.endoscopy_collate import endoscopy_stereo_collate
from eaglevision.data.endoscopy_stereo_dataset import EndoscopyStereoDataset
from eaglevision.utils.io import ensure_dir


CSV_COLUMNS = [
    "method",
    "sample_id",
    "sequence_id",
    "frame_id",
    "absrel",
    "sqrel",
    "rmse",
    "rmse_log",
    "silog",
    "delta1",
    "delta2",
    "delta3",
    "reproj_ms_ssim",
    "photo_l1",
    "valid_reproj_ratio",
    "disp_absrel",
    "disp_rmse",
    "disp_log_l1",
    "inference_ms",
    "peak_gpu_mem_mb",
]


def _device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("device", "cuda"))
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _dataset(config: dict[str, Any], manifest: Path) -> EndoscopyStereoDataset:
    data_cfg = config.get("data", {})
    size = data_cfg.get("image_size")
    return EndoscopyStereoDataset(
        manifest_path=manifest,
        root=Path(data_cfg["root"]) if data_cfg.get("root") else None,
        image_size=tuple(size) if size else None,
        min_depth=float(data_cfg.get("min_depth", 1e-3)),
        max_depth=float(data_cfg.get("max_depth", 300.0)),
    )


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _nan_depth() -> dict[str, float]:
    return {key: float("nan") for key in ("absrel", "sqrel", "rmse", "rmse_log", "silog", "delta1", "delta2", "delta3")}


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"num_samples": len(rows)}
    for key in CSV_COLUMNS[4:]:
        values = []
        for row in rows:
            value = row.get(key)
            if value in (None, ""):
                continue
            value_f = float(value)
            if math.isfinite(value_f):
                values.append(value_f)
        if values:
            summary[key] = sum(values) / len(values)
    return summary


def evaluate_dares(config: dict[str, Any], manifest: Path, checkpoint: Path, out_path: Path) -> None:
    device = _device(config)
    dataset = _dataset(config, manifest)
    eval_cfg = config.get("eval", {})
    loader = DataLoader(dataset, batch_size=int(eval_cfg.get("batch_size", 1)), shuffle=False, num_workers=int(eval_cfg.get("num_workers", 0)), collate_fn=endoscopy_stereo_collate)
    model = DARESDepthAnything(config).to(device)
    payload = torch.load(checkpoint, map_location="cpu")
    load_vector_lora_state_dict(model, payload["vector_lora"], strict=True)
    model.eval()
    rows: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    peak_gpu_mem_mb = float("nan")
    reproj_cfg = config.get("loss", {}).get("reprojection", {})
    scales = tuple(int(scale) for scale in reproj_cfg.get("scales", [1, 2, 4]))
    alpha = float(reproj_cfg.get("alpha", 0.85))
    min_depth = float(config.get("data", {}).get("min_depth", 1e-3))
    max_depth = float(config.get("data", {}).get("max_depth", 300.0))

    with torch.no_grad():
        for batch in tqdm(loader, desc="dares eval"):
            batch = _move(batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            depth = model.predict_depth(batch["left_rgb"])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_ms = (time.perf_counter() - start) * 1000.0 / max(depth.shape[0], 1)
            for index in range(depth.shape[0]):
                sample_depth = depth[index : index + 1]
                sample_batch = {
                    key: value[index : index + 1] if torch.is_tensor(value) and value.shape[:1] == depth.shape[:1] else value
                    for key, value in batch.items()
                }
                depth_metrics = _nan_depth()
                if batch.get("depth_gt") is not None:
                    depth_metrics = compute_depth_metrics(sample_depth, batch["depth_gt"][index : index + 1], min_depth=min_depth, max_depth=max_depth)
                reproj = {"reproj_ms_ssim": float("nan"), "photo_l1": float("nan"), "valid_reproj_ratio": float("nan"), "predicted_disparity": None}
                if sample_batch.get("focal_length") is not None and sample_batch.get("baseline") is not None:
                    reproj = reprojection_metrics(sample_batch["left_rgb"], sample_batch["right_rgb"], sample_depth, sample_batch["focal_length"], sample_batch["baseline"], scales=scales, alpha=alpha, min_depth=min_depth)
                disp = {"disp_absrel": float("nan"), "disp_rmse": float("nan"), "disp_log_l1": float("nan")}
                if sample_batch.get("disparity_gt") is not None and reproj.get("predicted_disparity") is not None:
                    disp = disparity_metrics(reproj["predicted_disparity"], sample_batch["disparity_gt"])
                row = {
                    "method": config.get("method", "dares_da2s_vector_lora"),
                    "sample_id": batch["sample_id"][index],
                    "sequence_id": batch["sequence_id"][index],
                    "frame_id": batch["frame_id"][index],
                    **depth_metrics,
                    "reproj_ms_ssim": reproj["reproj_ms_ssim"],
                    "photo_l1": reproj["photo_l1"],
                    "valid_reproj_ratio": reproj["valid_reproj_ratio"],
                    **disp,
                    "inference_ms": inference_ms,
                    "peak_gpu_mem_mb": peak_gpu_mem_mb,
                }
                rows.append(row)
    if device.type == "cuda":
        peak_gpu_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        for row in rows:
            row["peak_gpu_mem_mb"] = peak_gpu_mem_mb
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = _summarize(rows)
    summary["trainable_param_count"] = model.trainable_parameter_count()
    summary["vector_lora_summary"] = model.vector_lora_summary()
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md = [f"# {config.get('method', 'DARES')} Evaluation", "", f"- Samples: {len(rows)}", f"- Trainable params: {model.trainable_parameter_count()}"]
    for key in ("absrel", "rmse", "silog", "delta1", "reproj_ms_ssim", "photo_l1", "valid_reproj_ratio", "inference_ms"):
        value = summary.get(key, float("nan"))
        md.append(f"- {key}: {value:.6g}" if math.isfinite(float(value)) else f"- {key}: NaN")
    out_path.with_suffix(".summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
