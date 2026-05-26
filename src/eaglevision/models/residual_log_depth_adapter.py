from __future__ import annotations

import torch
from torch import nn


class ResidualLogDepthAdapter(nn.Module):
    """Tiny output adapter that predicts a bounded residual in log-depth space."""

    def __init__(
        self,
        hidden_channels: int = 32,
        residual_scale: float = 0.1,
        use_tanh: bool = True,
        min_depth: float = 1e-3,
        max_depth: float = 300.0,
    ) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.use_tanh = bool(use_tanh)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.net = nn.Sequential(
            nn.Conv2d(4, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, image: torch.Tensor, base_depth: torch.Tensor) -> dict[str, torch.Tensor]:
        if base_depth.ndim == 3:
            base_depth_chw = base_depth.unsqueeze(1)
        elif base_depth.ndim == 4 and base_depth.shape[1] == 1:
            base_depth_chw = base_depth
        else:
            raise ValueError(f"Expected base_depth shape [B,H,W] or [B,1,H,W], got {tuple(base_depth.shape)}")

        base_depth_chw = base_depth_chw.clamp_min(self.min_depth)
        log_base = base_depth_chw.log()
        residual = self.net(torch.cat((image, log_base), dim=1))
        if self.use_tanh:
            residual = residual.tanh()
        log_adapted = log_base + self.residual_scale * residual
        adapted_depth = log_adapted.exp().clamp(self.min_depth, self.max_depth)

        return {
            "base_depth": base_depth_chw.squeeze(1),
            "adapted_depth": adapted_depth.squeeze(1),
            "log_base_depth": log_base.squeeze(1),
            "log_adapted_depth": log_adapted.squeeze(1),
            "residual": residual,
        }
