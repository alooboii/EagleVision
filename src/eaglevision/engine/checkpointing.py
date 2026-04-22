from __future__ import annotations

from pathlib import Path

import torch

from eaglevision.utils.io import ensure_dir


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int) -> None:
    """Save a training checkpoint."""
    ensure_dir(path.parent)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, path)


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> int:
    """Load checkpoint state and return the stored global step."""
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("step", 0))
