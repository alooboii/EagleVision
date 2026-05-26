from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from eaglevision.dares.depth_anything_dares import DARESDepthAnything
from eaglevision.dares.losses import compute_dares_losses
from eaglevision.dares.vector_lora import list_trainable_parameters, load_vector_lora_state_dict, vector_lora_state_dict
from eaglevision.data.endoscopy_collate import endoscopy_stereo_collate
from eaglevision.data.endoscopy_stereo_dataset import EndoscopyStereoDataset
from eaglevision.utils.io import ensure_dir, save_yaml
from eaglevision.utils.seed import set_seed
from eaglevision.utils.visualization import depth_to_colormap, tensor_to_image


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


def model_outputs(model: DARESDepthAnything, batch: dict[str, Any], symmetric: bool = True) -> dict[str, torch.Tensor]:
    outputs = {"pred_depth_left": model(batch["left_rgb"])}
    if symmetric:
        outputs["pred_depth_right"] = model(batch["right_rgb"])
    return outputs


def save_panel(path: Path, batch: dict[str, Any], losses: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]) -> None:
    ensure_dir(path.parent)
    panels = [
        ("left_rgb", tensor_to_image(batch["left_rgb"][0])),
        ("right_rgb", tensor_to_image(batch["right_rgb"][0])),
        ("pred_depth_left", depth_to_colormap(outputs["pred_depth_left"][0])),
        ("pred_disparity_left", depth_to_colormap(losses["predicted_disparity_left"][0])),
        ("reconstructed_right", tensor_to_image(losses["right_reconstruction"][0])),
        ("valid_mask", losses["valid_reproj_mask"][0].detach().cpu().numpy()),
    ]
    if "left_reconstruction" in losses:
        panels.append(("left_reconstruction", tensor_to_image(losses["left_reconstruction"][0])))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 4))
    axes_list = list(axes) if hasattr(axes, "__iter__") else [axes]
    for axis, (title, image) in zip(axes_list, panels, strict=True):
        axis.imshow(image, cmap="gray" if getattr(image, "ndim", 3) == 2 else None)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_checkpoint(
    path: Path,
    model: DARESDepthAnything,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    config: dict[str, Any],
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "vector_lora": vector_lora_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "config": config,
            "vector_lora_summary": model.vector_lora_summary(),
        },
        path,
    )


@torch.no_grad()
def validate(model: DARESDepthAnything, loader: DataLoader, device: torch.device, config: dict[str, Any]) -> dict[str, float]:
    model.eval()
    records: list[dict[str, float]] = []
    symmetric = bool(config.get("loss", {}).get("reprojection", {}).get("symmetric", True))
    for batch in tqdm(loader, desc="dares val", leave=False):
        batch = _move(batch, device)
        outputs = model_outputs(model, batch, symmetric=symmetric)
        loss_cfg = {**config.get("loss", {}), "min_depth": config.get("data", {}).get("min_depth", 1e-3)}
        losses = compute_dares_losses(batch, outputs, loss_cfg)
        records.append({key: float(value.detach().cpu().item()) for key, value in losses.items() if key.startswith("loss_")})
    keys = sorted({key for record in records for key in record})
    return {key: sum(record.get(key, 0.0) for record in records) / max(len(records), 1) for key in keys}


def run_dry_run(config: dict[str, Any], train_manifest: Path, val_manifest: Path) -> dict[str, float]:
    set_seed(int(config.get("seed", 42)))
    device = _device(config)
    output_dir = ensure_dir(Path(config.get("output_dir", "outputs/dares/dares_da2s_scared")))
    train_cfg = config.get("train", {})
    loader = DataLoader(
        _dataset(config, train_manifest),
        batch_size=int(train_cfg.get("batch_size", 4)),
        shuffle=False,
        num_workers=0,
        collate_fn=endoscopy_stereo_collate,
    )
    _dataset(config, val_manifest)
    model = DARESDepthAnything(config).to(device)
    batch = _move(next(iter(loader)), device)
    symmetric = bool(config.get("loss", {}).get("reprojection", {}).get("symmetric", True))
    outputs = model_outputs(model, batch, symmetric=symmetric)
    loss_cfg = {**config.get("loss", {}), "min_depth": config.get("data", {}).get("min_depth", 1e-3)}
    losses = compute_dares_losses(batch, outputs, loss_cfg)
    losses["loss_total"].backward()
    trainable = {name for name in list_trainable_parameters(model)}
    lora_grad = sum(float(param.grad.detach().abs().sum().item()) for name, param in model.named_parameters() if name in trainable and param.grad is not None)
    frozen_grads = [name for name, param in model.named_parameters() if name not in trainable and param.grad is not None and float(param.grad.abs().sum().item()) > 0]
    if lora_grad <= 0:
        raise RuntimeError("DARES dry-run failed: no Vector-LoRA gradients.")
    if frozen_grads:
        raise RuntimeError(f"DARES dry-run failed: frozen base parameters received gradients: {frozen_grads[:10]}")
    save_panel(output_dir / "panels" / "dry_run.png", batch, losses, outputs)
    _save_checkpoint(output_dir / "checkpoints" / "dry_run.pt", model, torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4), None, 0, 0, float(losses["loss_total"].detach().cpu().item()), config)
    return {key: float(value.detach().cpu().item()) for key, value in losses.items() if key.startswith("loss_")} | {"vector_lora_grad_norm": lora_grad}


def train_dares(config: dict[str, Any], train_manifest: Path, val_manifest: Path, resume: Path | None = None) -> None:
    set_seed(int(config.get("seed", 42)))
    device = _device(config)
    output_dir = ensure_dir(Path(config.get("output_dir", "outputs/dares/dares_da2s_scared")))
    ensure_dir(output_dir / "checkpoints")
    ensure_dir(output_dir / "panels")
    save_yaml(output_dir / "config_used.yaml", config)

    train_cfg = config.get("train", {})
    train_loader = DataLoader(_dataset(config, train_manifest), batch_size=int(train_cfg.get("batch_size", 4)), shuffle=True, num_workers=int(train_cfg.get("num_workers", 2)), collate_fn=endoscopy_stereo_collate)
    val_loader = DataLoader(_dataset(config, val_manifest), batch_size=int(train_cfg.get("batch_size", 4)), shuffle=False, num_workers=int(train_cfg.get("num_workers", 2)), collate_fn=endoscopy_stereo_collate)
    model = DARESDepthAnything(config).to(device)
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(train_cfg.get("lr", 1e-4)), weight_decay=float(train_cfg.get("weight_decay", 1e-4)))
    scheduler = None
    amp_enabled = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    if resume:
        payload = torch.load(resume, map_location="cpu")
        load_vector_lora_state_dict(model, payload["vector_lora"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        global_step = int(payload.get("global_step", 0))
        best_val_loss = float(payload.get("best_val_loss", best_val_loss))

    (output_dir / "vector_lora_summary.json").write_text(json.dumps(model.vector_lora_summary(), indent=2) + "\n", encoding="utf-8")
    (output_dir / "param_summary.json").write_text(
        json.dumps({"trainable_param_count": model.trainable_parameter_count(), "total_param_count": model.total_parameter_count()}, indent=2) + "\n",
        encoding="utf-8",
    )
    loss_cfg = {**config.get("loss", {}), "min_depth": config.get("data", {}).get("min_depth", 1e-3)}
    max_steps = train_cfg.get("max_steps_per_epoch")
    max_steps_per_epoch = int(max_steps) if max_steps is not None else None
    grad_clip = train_cfg.get("grad_clip_norm")
    val_interval = int(train_cfg.get("val_interval", 1))
    log_interval = int(train_cfg.get("log_interval", 20))
    symmetric = bool(config.get("loss", {}).get("reprojection", {}).get("symmetric", True))
    metrics_path = output_dir / "train_metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as handle:
        for epoch in range(start_epoch, int(train_cfg.get("epochs", 20)) + 1):
            model.train()
            for step, batch in enumerate(tqdm(train_loader, desc=f"dares epoch {epoch}"), start=1):
                if max_steps_per_epoch is not None and step > max_steps_per_epoch:
                    break
                batch = _move(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    outputs = model_outputs(model, batch, symmetric=symmetric)
                    losses = compute_dares_losses(batch, outputs, loss_cfg)
                scaler.scale(losses["loss_total"]).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, float(grad_clip))
                scaler.step(optimizer)
                scaler.update()
                global_step += 1
                if step == 1:
                    save_panel(output_dir / "panels" / f"epoch_{epoch:03d}.png", batch, losses, outputs)
                record = {"split": "train", "epoch": epoch, "global_step": global_step, "lr": optimizer.param_groups[0]["lr"]}
                record.update({key: float(value.detach().cpu().item()) for key, value in losses.items() if key.startswith("loss_")})
                if global_step % log_interval == 0:
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
            if epoch % val_interval == 0:
                val_metrics = validate(model, val_loader, device, config)
                val_loss = float(val_metrics.get("loss_total", float("inf")))
                handle.write(json.dumps({"split": "val", "epoch": epoch, "global_step": global_step, **val_metrics}) + "\n")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    _save_checkpoint(output_dir / "checkpoints" / "best.pt", model, optimizer, scheduler, epoch, global_step, best_val_loss, config)
            _save_checkpoint(output_dir / "checkpoints" / "last.pt", model, optimizer, scheduler, epoch, global_step, best_val_loss, config)
