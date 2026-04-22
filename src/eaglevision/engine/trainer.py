from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from eaglevision.engine.checkpointing import save_checkpoint
from eaglevision.engine.logging import JsonlLogger
from eaglevision.engine.evaluator import evaluate_model
from eaglevision.losses.total import compute_phase1_losses


class Trainer:
    """Simple training loop for Phase 1 experiments."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        device: torch.device,
        output_dir: Path,
        loss_weights: dict[str, float],
        log_interval: int = 10,
        vis_interval: int = 100,
        checkpoint_interval: int = 500,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir
        self.loss_weights = loss_weights
        self.log_interval = log_interval
        self.vis_interval = vis_interval
        self.checkpoint_interval = checkpoint_interval
        self.logger = JsonlLogger(output_dir / "train_metrics.jsonl")

    def train(self, num_epochs: int) -> None:
        step = 0
        self.model.to(self.device)
        for epoch in range(num_epochs):
            self.model.train()
            for batch in tqdm(self.train_loader, desc=f"epoch {epoch + 1}/{num_epochs}"):
                batch = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in batch.items()}
                outputs = self.model(batch)
                losses = compute_phase1_losses(outputs, self.loss_weights)
                self.optimizer.zero_grad(set_to_none=True)
                losses["loss_total"].backward()
                self.optimizer.step()
                step += 1

                if step % self.log_interval == 0:
                    self.logger.log({"step": step, "epoch": epoch + 1, **{k: float(v.item()) for k, v in losses.items()}})
                if step % self.vis_interval == 0:
                    from eaglevision.visualization.save_panels import save_debug_panel

                    save_debug_panel(outputs, self.output_dir / "panels" / f"step_{step:07d}.png")
                if step % self.checkpoint_interval == 0:
                    save_checkpoint(self.output_dir / "checkpoints" / f"step_{step:07d}.pt", self.model, self.optimizer, step)

            if self.val_loader is not None:
                metrics = evaluate_model(self.model, self.val_loader, self.device, self.loss_weights)
                self.logger.log({"step": step, "epoch": epoch + 1, "split": "val", **metrics})
