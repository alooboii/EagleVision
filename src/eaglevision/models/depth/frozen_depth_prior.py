from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from baseline.depth_anything_v2.inference import _extract_state_dict
from baseline.depth_anything_v2.modeling import create_model


class FrozenDepthPrior(nn.Module):
    """Frozen Depth Anything V2 prior with the same padding behavior as the legacy wrapper."""

    def __init__(
        self,
        mode: str = "relative",
        encoder: str = "vits",
        profile: str = "hypersim",
        checkpoint_path: Path | None = None,
        normalize_backbone_input: bool = False,
        freeze: bool = True,
        force_eval: bool | None = None,
    ) -> None:
        super().__init__()
        self.backbone = create_model(mode=mode, encoder=encoder, profile=profile)
        if checkpoint_path is not None:
            checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")
            state_dict = _extract_state_dict(checkpoint_obj)
            self.backbone.load_state_dict(state_dict, strict=False)
        if freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            self.backbone.eval()
        self.force_eval = freeze if force_eval is None else bool(force_eval)
        self.mode = mode
        self.encoder = encoder
        self.profile = profile
        self.normalize_backbone_input = normalize_backbone_input
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))

    def train(self, mode: bool = True) -> "FrozenDepthPrior":
        super().train(False if self.force_eval else mode)
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_depth(images)

    def forward_depth(self, images: torch.Tensor) -> torch.Tensor:
        backbone_input = images
        if self.normalize_backbone_input:
            backbone_input = (backbone_input - self.rgb_mean) / self.rgb_std

        input_h, input_w = backbone_input.shape[-2:]
        pad_h = (14 - input_h % 14) % 14
        pad_w = (14 - input_w % 14) % 14
        if pad_h or pad_w:
            backbone_input = F.pad(backbone_input, (0, pad_w, 0, pad_h), mode="replicate")

        depth = self.backbone(backbone_input)
        if pad_h or pad_w:
            depth = depth[..., :input_h, :input_w]
        return depth

    @torch.no_grad()
    def predict_depth(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_depth(images)


def build_frozen_depth_prior(config: dict) -> FrozenDepthPrior:
    checkpoint_path = config.get("checkpoint_path")
    encoder = str(config.get("encoder", "vits"))
    encoder = {"small": "vits", "base": "vitb", "large": "vitl"}.get(encoder, encoder)
    return FrozenDepthPrior(
        mode=str(config.get("mode", "relative")),
        encoder=encoder,
        profile=str(config.get("profile", "hypersim")),
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
        normalize_backbone_input=bool(config.get("normalize_backbone_input", False)),
        freeze=bool(config.get("freeze", True)),
        force_eval=True,
    )


def build_trainable_depth_prior(config: dict) -> FrozenDepthPrior:
    checkpoint_path = config.get("checkpoint_path")
    encoder = str(config.get("encoder", "vits"))
    encoder = {"small": "vits", "base": "vitb", "large": "vitl"}.get(encoder, encoder)
    return FrozenDepthPrior(
        mode=str(config.get("mode", "relative")),
        encoder=encoder,
        profile=str(config.get("profile", "hypersim")),
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
        normalize_backbone_input=bool(config.get("normalize_backbone_input", False)),
        freeze=True,
        force_eval=False,
    )
