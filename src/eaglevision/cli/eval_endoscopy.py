from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from eaglevision.data.endoscopy_collate import endoscopy_stereo_collate
from eaglevision.data.endoscopy_stereo_dataset import EndoscopyStereoDataset
from eaglevision.geometry.stereo_warp import compute_disparity_from_depth, warp_image_with_disparity
from eaglevision.losses.endoscopy_cycle_losses import photometric_ssim_l1
from eaglevision.metrics.depth_metrics import compute_depth_metrics
from eaglevision.models.depth.frozen_depth_prior import build_frozen_depth_prior, build_trainable_depth_prior
from eaglevision.models.peft_lora_da2 import apply_lora_to_named_linears
from eaglevision.models.residual_log_depth_adapter import ResidualLogDepthAdapter
from eaglevision.utils.io import ensure_dir, load_yaml


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
    "photo_l1",
    "photo_ssim_l1",
    "cycle_rgb_l1",
    "valid_reproj_ratio",
    "disp_absrel",
    "disp_rmse",
    "inference_ms",
    "peak_gpu_mem_mb",
]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate frozen/adapted DA2 on endoscopy stereo manifests.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def _device_from_config(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("device", "cuda"))
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _build_dataset(config: dict[str, Any], manifest: Path) -> EndoscopyStereoDataset:
    data_cfg = config.get("data", {})
    image_size = data_cfg.get("image_size")
    return EndoscopyStereoDataset(
        manifest_path=manifest,
        root=Path(data_cfg["root"]) if data_cfg.get("root") else None,
        image_size=tuple(image_size) if image_size is not None else None,
        min_depth=float(data_cfg.get("min_depth", 1e-3)),
        max_depth=float(data_cfg.get("max_depth", 300.0)),
    )


def _load_adapter(config: dict[str, Any], checkpoint: Path | None, device: torch.device) -> ResidualLogDepthAdapter | None:
    if checkpoint is None:
        return None
    adapter = ResidualLogDepthAdapter(**config.get("adapter", {})).to(device)
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("adapter", payload)
    adapter.load_state_dict(state)
    adapter.eval()
    return adapter


def _apply_lora_from_config(model: torch.nn.Module, config: dict[str, Any]) -> dict[str, object]:
    lora_cfg = config.get("lora", {})
    return apply_lora_to_named_linears(
        model,
        target_substrings=list(lora_cfg.get("target_substrings", ["attn", "q", "v", "proj"])),
        rank=int(lora_cfg.get("rank", 4)),
        alpha=float(lora_cfg.get("alpha", 8.0)),
        dropout=float(lora_cfg.get("dropout", 0.0)),
    )


def _load_lora_prior(config: dict[str, Any], checkpoint: Path | None, device: torch.device) -> tuple[torch.nn.Module, dict[str, object]]:
    prior = build_trainable_depth_prior(config.get("base_model", {})).to(device)
    summary = _apply_lora_from_config(prior.backbone, config)
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu")
        state = payload.get("lora", payload.get("model", payload))
        missing, unexpected = prior.backbone.load_state_dict(state, strict=False)
        unexpected_lora = [name for name in unexpected if "lora_" in name]
        if unexpected_lora:
            raise RuntimeError(f"Unexpected LoRA checkpoint keys: {unexpected_lora[:10]}")
        summary = {**summary, "checkpoint_missing_keys": len(missing), "checkpoint_unexpected_keys": len(unexpected)}
    prior.eval()
    return prior, summary


def _nan_metrics() -> dict[str, float]:
    return {key: float("nan") for key in ("absrel", "sqrel", "rmse", "rmse_log", "silog", "delta1", "delta2", "delta3")}


def _disparity_metrics(predicted_disparity: torch.Tensor | None, gt_disparity: torch.Tensor | None) -> dict[str, float]:
    if predicted_disparity is None or gt_disparity is None:
        return {"disp_absrel": float("nan"), "disp_rmse": float("nan")}
    pred = predicted_disparity.squeeze(1) if predicted_disparity.ndim == 4 and predicted_disparity.shape[1] == 1 else predicted_disparity
    target = gt_disparity.to(pred.device)
    target = target.squeeze(1) if target.ndim == 4 and target.shape[1] == 1 else target
    valid = torch.isfinite(pred) & torch.isfinite(target) & (pred > 0) & (target > 0)
    if not valid.any():
        return {"disp_absrel": float("nan"), "disp_rmse": float("nan")}
    diff = pred[valid] - target[valid]
    return {
        "disp_absrel": float((diff.abs() / target[valid].clamp_min(1e-6)).mean().detach().cpu().item()),
        "disp_rmse": float(torch.sqrt((diff**2).mean()).detach().cpu().item()),
    }


def _geometry_metrics(batch: dict[str, Any], prediction: torch.Tensor, config: dict[str, Any]) -> dict[str, float]:
    left = batch["left_rgb"]
    right = batch["right_rgb"]
    disparity = None
    if batch.get("focal_length") is not None and batch.get("baseline") is not None:
        disparity = compute_disparity_from_depth(
            prediction,
            batch["focal_length"],
            batch["baseline"],
            min_depth=float(config.get("data", {}).get("min_depth", 1e-3)),
        )
    disp_metrics = _disparity_metrics(disparity, batch.get("disparity_gt"))
    if disparity is None:
        return {
            "photo_l1": float("nan"),
            "photo_ssim_l1": float("nan"),
            "cycle_rgb_l1": float("nan"),
            "valid_reproj_ratio": float("nan"),
            **disp_metrics,
        }

    reproj = warp_image_with_disparity(left, disparity, direction="left_to_right")
    mask = reproj["valid_mask"]
    if not mask.any():
        return {
            "photo_l1": float("nan"),
            "photo_ssim_l1": float("nan"),
            "cycle_rgb_l1": float("nan"),
            "valid_reproj_ratio": 0.0,
            **disp_metrics,
        }
    photo_l1 = torch.abs(reproj["warped"] - right).mean(dim=1)[mask].mean()
    photo_ssim_l1 = photometric_ssim_l1(reproj["warped"], right, mask=mask, alpha=float(config.get("loss", {}).get("photometric", {}).get("alpha", 0.85)))
    cycle = warp_image_with_disparity(reproj["warped"], disparity, direction="right_to_left")
    cycle_mask = mask & cycle["valid_mask"]
    cycle_rgb_l1 = torch.abs(cycle["warped"] - left).mean(dim=1)[cycle_mask].mean() if cycle_mask.any() else torch.tensor(float("nan"), device=left.device)
    return {
        "photo_l1": float(photo_l1.detach().cpu().item()),
        "photo_ssim_l1": float(photo_ssim_l1.detach().cpu().item()),
        "cycle_rgb_l1": float(cycle_rgb_l1.detach().cpu().item()),
        "valid_reproj_ratio": float(mask.float().mean().detach().cpu().item()),
        **disp_metrics,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"num_samples": len(rows)}
    for key in CSV_COLUMNS[4:]:
        values = []
        for row in rows:
            if key not in row or row[key] in ("", None):
                continue
            value = float(row[key])
            if math.isfinite(value):
                values.append(value)
        if values:
            summary[key] = sum(values) / len(values)
    return summary


def main() -> int:
    args = build_argparser().parse_args()
    config = load_yaml(args.config)
    device = _device_from_config(config)
    method = str(config.get("method", "eaglevision_da2s_cycle" if args.checkpoint else "frozen_da2s"))

    dataset = _build_dataset(config, args.manifest)
    eval_cfg = config.get("eval", {})
    loader = DataLoader(
        dataset,
        batch_size=int(eval_cfg.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(eval_cfg.get("num_workers", 0)),
        collate_fn=endoscopy_stereo_collate,
    )

    lora_summary: dict[str, object] | None = None
    if method == "dares_style_lora_da2s_cycle":
        prior, lora_summary = _load_lora_prior(config, args.checkpoint, device)
        adapter = None
    else:
        prior = build_frozen_depth_prior(config.get("base_model", {})).to(device)
        prior.eval()
        adapter = _load_adapter(config, args.checkpoint, device) if method == "eaglevision_da2s_cycle" else None

    rows: list[dict[str, Any]] = []
    peak_gpu_mem_mb = float("nan")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval endoscopy"):
            batch = _move_batch(batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            if method == "dares_style_lora_da2s_cycle":
                base_depth = prior.forward_depth(batch["left_rgb"])
                prediction = base_depth
            else:
                base_depth = prior.predict_depth(batch["left_rgb"])
            if adapter is not None:
                adapter_outputs = adapter(batch["left_rgb"], base_depth)
                prediction = adapter_outputs["adapted_depth"]
            elif method != "dares_style_lora_da2s_cycle":
                prediction = base_depth
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_ms = (time.perf_counter() - start) * 1000.0 / max(prediction.shape[0], 1)

            for index in range(prediction.shape[0]):
                sample_batch = {
                    key: value[index : index + 1] if torch.is_tensor(value) and value.shape[:1] == prediction.shape[:1] else value
                    for key, value in batch.items()
                }
                sample_pred = prediction[index : index + 1]
                if batch.get("depth_gt") is not None:
                    depth_metrics = compute_depth_metrics(
                        sample_pred,
                        batch["depth_gt"][index : index + 1],
                        min_depth=float(config.get("data", {}).get("min_depth", 1e-3)),
                        max_depth=float(config.get("data", {}).get("max_depth", 300.0)),
                    )
                else:
                    depth_metrics = _nan_metrics()
                geom_metrics = _geometry_metrics(sample_batch, sample_pred, config)
                row = {
                    "method": method,
                    "sample_id": batch["sample_id"][index],
                    "sequence_id": batch["sequence_id"][index],
                    "frame_id": batch["frame_id"][index],
                    **depth_metrics,
                    **geom_metrics,
                    "inference_ms": inference_ms,
                    "peak_gpu_mem_mb": peak_gpu_mem_mb,
                }
                rows.append(row)

    if device.type == "cuda":
        peak_gpu_mem_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        for row in rows:
            row["peak_gpu_mem_mb"] = peak_gpu_mem_mb

    ensure_dir(args.out.parent)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    summary = _summarize(rows)
    if lora_summary is not None:
        summary["lora"] = lora_summary
        summary["trainable_param_count"] = lora_summary.get("trainable_lora_params")
    summary_path = args.out.with_suffix(".summary.json")
    md_path = args.out.with_suffix(".summary.md")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md_lines = [f"# {method} Endoscopy Evaluation", "", f"- Samples: {len(rows)}"]
    for key in ("absrel", "rmse", "silog", "photo_l1", "photo_ssim_l1", "cycle_rgb_l1", "disp_absrel", "disp_rmse", "inference_ms"):
        value = summary.get(key, float("nan"))
        md_lines.append(f"- {key}: {value:.6g}" if math.isfinite(float(value)) else f"- {key}: NaN")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote per-sample metrics to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
