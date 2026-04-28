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
        normalize_backbone_input: bool = False,
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
        self.normalize_backbone_input = normalize_backbone_input
        # DA-V2 expects ImageNet-style normalized RGB at inference.
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        backbone_input = images
        if self.normalize_backbone_input:
            backbone_input = (backbone_input - self.rgb_mean) / self.rgb_std

        # The baseline prefers inputs divisible by 14; keep the resize local to the wrapper.
        input_h, input_w = backbone_input.shape[-2:]
        pad_h = (14 - input_h % 14) % 14
        pad_w = (14 - input_w % 14) % 14
        if pad_h or pad_w:
            resized = F.pad(backbone_input, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            resized = backbone_input
        with torch.set_grad_enabled(any(parameter.requires_grad for parameter in self.backbone.parameters())):
            base_depth = self.backbone(resized)
        if pad_h or pad_w:
            base_depth = base_depth[..., :input_h, :input_w]
        adapted_depth = self.adapter(images, base_depth)
        return {"base_depth": base_depth, "adapted_depth": adapted_depth}
