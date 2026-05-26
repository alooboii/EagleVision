from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from eaglevision.cli.train_endoscopy_adapter import _build_dataset, _device_from_config, _mean_metrics, _move_batch, _save_panel
from eaglevision.data.endoscopy_collate import endoscopy_stereo_collate
from eaglevision.losses.endoscopy_cycle_losses import compute_endoscopy_losses
from eaglevision.models.depth.frozen_depth_prior import build_trainable_depth_prior
from eaglevision.models.peft_lora_da2 import apply_lora_to_named_linears, lora_state_dict
from eaglevision.utils.io import ensure_dir, load_yaml, save_yaml
from eaglevision.utils.seed import set_seed


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DARES-style LoRA baseline on endoscopy stereo pairs.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    return parser


def _depth_outputs(depth: torch.Tensor) -> dict[str, torch.Tensor]:
    depth = depth.clamp_min(1e-3)
    log_depth = depth.log()
    return {
        "base_depth": depth,
        "adapted_depth": depth,
        "log_base_depth": log_depth,
        "log_adapted_depth": log_depth,
        "residual": torch.zeros(depth.shape[0], 1, depth.shape[1], depth.shape[2], device=depth.device, dtype=depth.dtype),
    }


def _build_lora_prior(config: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, dict[str, object]]:
    prior = build_trainable_depth_prior(config.get("base_model", {})).to(device)
    lora_cfg = config.get("lora", {})
    summary = apply_lora_to_named_linears(
        prior.backbone,
        target_substrings=list(lora_cfg.get("target_substrings", ["attn", "q", "v", "proj"])),
        rank=int(lora_cfg.get("rank", 4)),
        alpha=float(lora_cfg.get("alpha", 8.0)),
        dropout=float(lora_cfg.get("dropout", 0.0)),
    )
    if not summary["replaced_modules"]:
        raise RuntimeError("No DA2 Linear modules matched LoRA target_substrings.")
    return prior, summary


def _save_lora_checkpoint(
    path: Path,
    prior: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    val_loss: float,
    lora_summary: dict[str, object],
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "lora": lora_state_dict(prior.backbone),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "val_loss_total": val_loss,
            "lora_summary": lora_summary,
        },
        path,
    )


@torch.no_grad()
def validate_lora(
    prior: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_cfg: dict[str, Any],
) -> dict[str, float]:
    prior.eval()
    metrics: list[dict[str, float]] = []
    for batch in tqdm(loader, desc="val", leave=False):
        batch = _move_batch(batch, device)
        outputs = _depth_outputs(prior.forward_depth(batch["left_rgb"]))
        losses = compute_endoscopy_losses(batch, outputs, loss_cfg)
        metrics.append({key: float(value.detach().cpu().item()) for key, value in losses.items()})
    return _mean_metrics(metrics)


def main() -> int:
    args = build_argparser().parse_args()
    config = load_yaml(args.config)
    set_seed(int(config.get("seed", 42)))
    device = _device_from_config(config)

    output_dir = ensure_dir(Path(config.get("output_dir", "outputs/endo/lora_da2s_cycle_r4")))
    ensure_dir(output_dir / "checkpoints")
    ensure_dir(output_dir / "panels")
    save_yaml(output_dir / "config_used.yaml", config)

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

    prior, lora_summary = _build_lora_prior(config, device)
    trainable_params = [parameter for parameter in prior.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    param_summary = {
        **lora_summary,
        "trainable_param_count": sum(parameter.numel() for parameter in trainable_params),
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
            prior.train()
            epoch_metrics: list[dict[str, float]] = []
            progress = tqdm(train_loader, desc=f"lora train epoch {epoch}")
            for step, batch in enumerate(progress, start=1):
                if max_steps_per_epoch is not None and step > max_steps_per_epoch:
                    break
                batch = _move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                outputs = _depth_outputs(prior.forward_depth(batch["left_rgb"]))
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

            val_mean = validate_lora(prior, val_loader, device, config.get("loss", {})) if epoch % val_interval == 0 else {}
            if val_mean:
                metrics_file.write(json.dumps({"split": "val", "epoch": epoch, "step": global_step, **val_mean}) + "\n")
                val_loss = float(val_mean.get("loss_total", float("inf")))
                if val_loss < best_loss:
                    best_loss = val_loss
                    _save_lora_checkpoint(output_dir / "checkpoints" / "best.pt", prior, optimizer, epoch, global_step, val_loss, lora_summary)
            _save_lora_checkpoint(
                output_dir / "checkpoints" / "last.pt",
                prior,
                optimizer,
                epoch,
                global_step,
                float(val_mean.get("loss_total", best_loss)),
                lora_summary,
            )
            metrics_file.flush()

    print(f"Wrote DARES-style LoRA training outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
