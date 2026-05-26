from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from baseline.depth_anything_v2.inference import _extract_state_dict
from baseline.depth_anything_v2.modeling import create_model
from eaglevision.dares.vector_lora import (
    apply_vector_lora,
    count_trainable_parameters,
    freeze_non_vector_lora_parameters,
    list_trainable_parameters,
)


def _encoder_name(value: str) -> str:
    return {"small": "vits", "base": "vitb", "large": "vitl"}.get(value, value)


class DARESDepthAnything(nn.Module):
    """Depth Anything V2 with DARES Vector-LoRA adaptation."""

    def __init__(self, config: dict[str, Any], backbone: nn.Module | None = None) -> None:
        super().__init__()
        base_cfg = config.get("base_model", {})
        vector_cfg = config.get("vector_lora", {})
        if backbone is None:
            self.backbone = create_model(
                mode=str(base_cfg.get("mode", "relative")),
                encoder=_encoder_name(str(base_cfg.get("encoder", "small"))),
                profile=str(base_cfg.get("profile", "hypersim")),
            )
            checkpoint_path = base_cfg.get("checkpoint_path")
            if checkpoint_path:
                checkpoint_obj = torch.load(Path(checkpoint_path), map_location="cpu")
                self.backbone.load_state_dict(_extract_state_dict(checkpoint_obj), strict=False)
        else:
            self.backbone = backbone

        self.normalize_backbone_input = bool(base_cfg.get("normalize_backbone_input", False))
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))
        self._vector_lora_summary = apply_vector_lora(
            self.backbone,
            target_patterns=list(vector_cfg.get("target_patterns", ["qkv", "proj", "fc1", "fc2"])),
            schedule_config=vector_cfg,
            dropout=float(vector_cfg.get("dropout", 0.0)),
            include_modules=vector_cfg.get("include_modules"),
            exclude_modules=vector_cfg.get("exclude_modules"),
        )
        freeze_non_vector_lora_parameters(self.backbone)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
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

    def predict_depth(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward(images)

    def trainable_parameter_count(self) -> int:
        return count_trainable_parameters(self)

    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def vector_lora_summary(self) -> dict[str, Any]:
        return self._vector_lora_summary

    def trainable_parameter_names(self) -> list[str]:
        return list_trainable_parameters(self)
