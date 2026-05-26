from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from eaglevision.data.endoscopy_collate import endoscopy_stereo_collate
from eaglevision.data.endoscopy_stereo_dataset import EndoscopyStereoDataset
from eaglevision.geometry.stereo_warp import compute_disparity_from_depth, warp_image_with_disparity
from eaglevision.losses.endoscopy_cycle_losses import compute_endoscopy_losses
from eaglevision.models.depth.frozen_depth_prior import build_frozen_depth_prior
from eaglevision.models.residual_log_depth_adapter import ResidualLogDepthAdapter
from eaglevision.utils.io import ensure_dir, load_yaml, save_yaml
from eaglevision.utils.seed import set_seed
from eaglevision.utils.visualization import depth_to_colormap, tensor_to_image


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train frozen DA2 + output residual adapter on endoscopy stereo pairs.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
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


def _make_outputs(prior: torch.nn.Module, adapter: ResidualLogDepthAdapter, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        base_depth = prior.predict_depth(batch["left_rgb"])
    return adapter(batch["left_rgb"], base_depth)


def _mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted({key for item in items for key in item})
    return {key: sum(item.get(key, 0.0) for item in items) / len(items) for key in keys}


def _save_panel(path: Path, batch: dict[str, Any], outputs: dict[str, torch.Tensor]) -> None:
    ensure_dir(path.parent)
    panels = [
        ("left", tensor_to_image(batch["left_rgb"][0])),
        ("right", tensor_to_image(batch["right_rgb"][0])),
        ("base", depth_to_colormap(outputs["base_depth"][0])),
        ("adapted", depth_to_colormap(outputs["adapted_depth"][0])),
    ]
    with torch.no_grad():
        if batch.get("focal_length") is not None and batch.get("baseline") is not None:
            disparity = compute_disparity_from_depth(outputs["adapted_depth"], batch["focal_length"], batch["baseline"])
            right_recon = warp_image_with_disparity(batch["left_rgb"], disparity, direction="left_to_right")
            left_cycle = warp_image_with_disparity(right_recon["warped"], disparity, direction="right_to_left")
            panels.extend(
                [
                    ("pred_disp", depth_to_colormap(disparity[0])),
                    ("right_recon", tensor_to_image(right_recon["warped"][0])),
                    ("left_cycle", tensor_to_image(left_cycle["warped"][0])),
                ]
            )
    fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 4))
    axes_list = list(axes) if hasattr(axes, "__iter__") else [axes]
    for axis, (title, image) in zip(axes_list, panels, strict=True):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_checkpoint(path: Path, adapter: ResidualLogDepthAdapter, optimizer: torch.optim.Optimizer, epoch: int, global_step: int, val_loss: float) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "adapter": adapter.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "val_loss_total": val_loss,
        },
        path,
    )


def _adapter_grad_norm(adapter: ResidualLogDepthAdapter) -> float:
    total = 0.0
    for parameter in adapter.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().abs().sum().cpu().item())
    return total


def run_dry_run(
    config: dict[str, Any],
    train_manifest: Path,
    val_manifest: Path,
    device: torch.device,
    output_dir: Path,
    prior_builder=None,
) -> dict[str, float]:
    if prior_builder is None:
        prior_builder = build_frozen_depth_prior
    train_dataset = _build_dataset(config, train_manifest)
    _build_dataset(config, val_manifest)
    train_cfg = config.get("train", {})
    loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get("batch_size", 4)),
        shuffle=False,
        num_workers=0,
        collate_fn=endoscopy_stereo_collate,
    )
    batch = _move_batch(next(iter(loader)), device)
    prior = prior_builder(config.get("base_model", {})).to(device)
    adapter = ResidualLogDepthAdapter(**config.get("adapter", {})).to(device)
    for parameter in prior.parameters():
        parameter.requires_grad = False

    adapter.train()
    outputs = _make_outputs(prior, adapter, batch)
    losses = compute_endoscopy_losses(batch, outputs, config.get("loss", {}))
    adapter.zero_grad(set_to_none=True)
    losses["loss_total"].backward()
    grad_norm = _adapter_grad_norm(adapter)
    if grad_norm <= 0.0:
        raise RuntimeError("Dry-run failed: loss_total did not produce gradients for adapter parameters.")
    _save_panel(output_dir / "panels" / "dry_run.png", batch, outputs)
    return {**{key: float(value.detach().cpu().item()) for key, value in losses.items()}, "adapter_grad_norm": grad_norm}


@torch.no_grad()
def validate(
    prior: torch.nn.Module,
    adapter: ResidualLogDepthAdapter,
    loader: DataLoader,
    device: torch.device,
    loss_cfg: dict[str, Any],
) -> dict[str, float]:
    prior.eval()
    adapter.eval()
    metrics: list[dict[str, float]] = []
    for batch in tqdm(loader, desc="val", leave=False):
        batch = _move_batch(batch, device)
        outputs = _make_outputs(prior, adapter, batch)
        losses = compute_endoscopy_losses(batch, outputs, loss_cfg)
        metrics.append({key: float(value.detach().cpu().item()) for key, value in losses.items()})
    return _mean_metrics(metrics)


def main() -> int:
    args = build_argparser().parse_args()
    config = load_yaml(args.config)
    set_seed(int(config.get("seed", 42)))
    device = _device_from_config(config)

    output_dir = ensure_dir(Path(config.get("output_dir", "outputs/endo/eaglevision_da2s_cycle")))
    ensure_dir(output_dir / "checkpoints")
    ensure_dir(output_dir / "panels")
    save_yaml(output_dir / "config_used.yaml", config)

    if args.dry_run:
        metrics = run_dry_run(config, args.train_manifest, args.val_manifest, device, output_dir)
        (output_dir / "dry_run_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"dry_run": "ok", **metrics}, indent=2))
        return 0

    train_dataset = _build_dataset(config, args.train_manifest)
    val_dataset = _build_dataset(config, args.val_manifest)
    train_cfg = config.get("train", {})
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 2)),
        collate_fn=endoscopy_stereo_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg.get("batch_size", 4)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 2)),
        collate_fn=endoscopy_stereo_collate,
    )

    prior = build_frozen_depth_prior(config.get("base_model", {})).to(device)
    adapter = ResidualLogDepthAdapter(**config.get("adapter", {})).to(device)
    for parameter in prior.parameters():
        parameter.requires_grad = False

    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    param_summary = {
        "base_trainable": sum(parameter.numel() for parameter in prior.parameters() if parameter.requires_grad),
        "base_total": sum(parameter.numel() for parameter in prior.parameters()),
        "adapter_trainable": sum(parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad),
        "adapter_total": sum(parameter.numel() for parameter in adapter.parameters()),
    }
    (output_dir / "param_summary.json").write_text(json.dumps(param_summary, indent=2) + "\n", encoding="utf-8")

    metrics_path = output_dir / "train_metrics.jsonl"
    best_loss = float("inf")
    global_step = 0
    max_steps = train_cfg.get("max_steps_per_epoch")
    max_steps_per_epoch = int(max_steps) if max_steps is not None else None
    val_interval = int(train_cfg.get("val_interval", 1))

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(1, int(train_cfg.get("epochs", 5)) + 1):
            adapter.train()
            epoch_metrics: list[dict[str, float]] = []
            progress = tqdm(train_loader, desc=f"train epoch {epoch}")
            for step, batch in enumerate(progress, start=1):
                if max_steps_per_epoch is not None and step > max_steps_per_epoch:
                    break
                batch = _move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                outputs = _make_outputs(prior, adapter, batch)
                losses = compute_endoscopy_losses(batch, outputs, config.get("loss", {}))
                losses["loss_total"].backward()
                optimizer.step()
                global_step += 1
                loss_record = {key: float(value.detach().cpu().item()) for key, value in losses.items()}
                epoch_metrics.append(loss_record)
                if step == 1:
                    _save_panel(output_dir / "panels" / f"epoch_{epoch:03d}.png", batch, outputs)
                progress.set_postfix(loss=f"{loss_record['loss_total']:.4f}")
                if global_step % int(train_cfg.get("log_interval", 20)) == 0:
                    metrics_file.write(json.dumps({"split": "train", "epoch": epoch, "step": global_step, **loss_record}) + "\n")
                    metrics_file.flush()

            train_mean = _mean_metrics(epoch_metrics)
            metrics_file.write(json.dumps({"split": "train_epoch", "epoch": epoch, "step": global_step, **train_mean}) + "\n")

            val_mean = validate(prior, adapter, val_loader, device, config.get("loss", {})) if epoch % val_interval == 0 else {}
            if val_mean:
                metrics_file.write(json.dumps({"split": "val", "epoch": epoch, "step": global_step, **val_mean}) + "\n")
                val_loss = float(val_mean.get("loss_total", float("inf")))
                if val_loss < best_loss:
                    best_loss = val_loss
                    _save_checkpoint(output_dir / "checkpoints" / "best.pt", adapter, optimizer, epoch, global_step, val_loss)
            _save_checkpoint(output_dir / "checkpoints" / "last.pt", adapter, optimizer, epoch, global_step, float(val_mean.get("loss_total", best_loss)))
            metrics_file.flush()

    print(f"Wrote training outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
