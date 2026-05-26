from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Minimal LoRA adapter for an existing frozen nn.Linear layer."""

    def __init__(self, linear: nn.Linear, rank: int = 4, alpha: float = 8.0, dropout: float = 0.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.linear = linear
        for parameter in self.linear.parameters():
            parameter.requires_grad = False
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Linear(linear.in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, linear.out_features, bias=False)
        nn.init.normal_(self.lora_a.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.scale * self.lora_b(self.lora_a(self.dropout(x)))


@dataclass
class LoRAApplySummary:
    replaced_modules: list[str]
    trainable_lora_params: int
    frozen_base_params: int
    total_params: int

    def to_dict(self) -> dict[str, object]:
        return {
            "replaced_modules": self.replaced_modules,
            "trainable_lora_params": self.trainable_lora_params,
            "frozen_base_params": self.frozen_base_params,
            "total_params": self.total_params,
        }


def _matches(name: str, target_substrings: list[str]) -> bool:
    lowered = name.lower()
    return any(token.lower() in lowered for token in target_substrings)


def _get_parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def freeze_non_lora_parameters(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_a" in name or "lora_b" in name


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items() if ".lora_a." in name or ".lora_b." in name}


def apply_lora_to_named_linears(
    model: nn.Module,
    target_substrings: list[str],
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
) -> dict[str, object]:
    if not target_substrings:
        raise ValueError("target_substrings must contain at least one module-name substring")

    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and not isinstance(module, LoRALinear) and _matches(name, target_substrings)
    ]
    replaced: list[str] = []
    for name, module in candidates:
        parent, child_name = _get_parent_module(model, name)
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        replaced.append(name)

    freeze_non_lora_parameters(model)
    trainable_lora = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    frozen = total - trainable_lora
    return LoRAApplySummary(
        replaced_modules=replaced,
        trainable_lora_params=trainable_lora,
        frozen_base_params=frozen,
        total_params=total,
    ).to_dict()
