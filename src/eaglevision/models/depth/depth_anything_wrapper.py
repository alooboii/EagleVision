from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from baseline.depth_anything_v2.inference import _extract_state_dict
from baseline.depth_anything_v2.modeling import create_model

from eaglevision.models.adapters import ResidualDepthAdapter


class DepthAnythingWithAdapter(nn.Module):
    """Vendor-baseline wrapper with a light residual adaptation head."""

    def __init__(
        self,
        mode: str,
        encoder: str,
        profile: str,
        checkpoint_path: Path | None = None,
        freeze_backbone: bool = True,
        adapter_hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        self.backbone = create_model(mode=mode, encoder=encoder, profile=profile)
        if checkpoint_path is not None:
            checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")
            state_dict = _extract_state_dict(checkpoint_obj)
            self.backbone.load_state_dict(state_dict, strict=False)
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        self.adapter = ResidualDepthAdapter(hidden_channels=adapter_hidden_channels)
        self.mode = mode

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        # The baseline prefers inputs divisible by 14; keep the resize local to the wrapper.
        input_h, input_w = images.shape[-2:]
        pad_h = (14 - input_h % 14) % 14
        pad_w = (14 - input_w % 14) % 14
        if pad_h or pad_w:
            resized = F.pad(images, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            resized = images
        with torch.set_grad_enabled(any(parameter.requires_grad for parameter in self.backbone.parameters())):
            base_depth = self.backbone(resized)
        if pad_h or pad_w:
            base_depth = base_depth[..., :input_h, :input_w]
        adapted_depth = self.adapter(images, base_depth)
        return {"base_depth": base_depth, "adapted_depth": adapted_depth}
