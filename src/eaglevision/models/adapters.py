from __future__ import annotations

import torch
from torch import nn


class ResidualDepthAdapter(nn.Module):
    """Small trainable residual head over RGB and baseline depth."""

    def __init__(self, hidden_channels: int = 32, clamp_min: float = 1e-3) -> None:
        super().__init__()
        self.clamp_min = clamp_min
        self.net = nn.Sequential(
            nn.Conv2d(4, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, image: torch.Tensor, base_depth: torch.Tensor) -> torch.Tensor:
        residual = self.net(torch.cat((image, base_depth.unsqueeze(1)), dim=1)).squeeze(1)
        return (base_depth + residual).clamp_min(self.clamp_min)
