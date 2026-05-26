from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


class VectorLoRALinear(nn.Module):
    """Vector-LoRA wrapper for a frozen linear projection."""

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        module_name: str = "",
        block_index: int | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"Vector-LoRA rank must be positive, got {rank}")
        self.linear = linear
        for parameter in self.linear.parameters():
            parameter.requires_grad = False
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.module_name = module_name
        self.block_index = block_index
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(linear.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.scale * self.lora_B(self.lora_A(self.dropout(x)))


@dataclass
class VectorLoRASummary:
    replaced_modules: list[dict[str, Any]]
    skipped_modules: list[dict[str, Any]]
    trainable_vector_lora_params: int
    frozen_base_params: int
    total_params: int
    rank_by_module: dict[str, int]
    rank_by_block: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "replaced_modules": self.replaced_modules,
            "skipped_modules": self.skipped_modules,
            "trainable_vector_lora_params": self.trainable_vector_lora_params,
            "frozen_base_params": self.frozen_base_params,
            "total_params": self.total_params,
            "rank_by_module": self.rank_by_module,
            "rank_by_block": self.rank_by_block,
        }


def infer_block_index(module_name: str) -> int | None:
    match = re.search(r"(?:^|\\.)(?:blocks|block)\\.(\\d+)(?:\\.|$)", module_name)
    return int(match.group(1)) if match else None


def _explicit_mapping(config: dict[str, Any], key: str) -> dict[int, int] | None:
    value = config.get(key)
    if not value:
        return None
    return {int(block): int(rank) for block, rank in dict(value).items()}


def _scheduled_value(block_index: int | None, num_blocks: int, config: dict[str, Any], value_name: str) -> int:
    override = _explicit_mapping(config, f"per_block_{value_name}")
    if override is not None and block_index is not None:
        if block_index in override:
            return override[block_index]
        raise ValueError(f"per_block_{value_name} does not define block {block_index}")

    default = int(config.get(f"default_{value_name}", config.get(f"late_{value_name}", 4 if value_name == "rank" else 8)))
    if block_index is None or num_blocks <= 0:
        return default

    schedule_type = str(config.get("schedule_type", "decreasing"))
    if schedule_type != "decreasing":
        return default

    early = int(config.get(f"early_{value_name}", 16 if value_name == "rank" else 32))
    middle = int(config.get(f"middle_{value_name}", 8 if value_name == "rank" else 16))
    late = int(config.get(f"late_{value_name}", 4 if value_name == "rank" else 8))
    final = int(config.get(f"final_{value_name}", 2 if value_name == "rank" else 4))
    frac = block_index / max(num_blocks - 1, 1)
    if frac < 0.25:
        return early
    if frac < 0.5:
        return middle
    if frac < 0.75:
        return late
    return final


def build_rank_schedule(num_blocks: int, config: dict[str, Any]) -> dict[int, int]:
    override = _explicit_mapping(config, "per_block_rank")
    if override is not None:
        return override
    return {block: _scheduled_value(block, num_blocks, config, "rank") for block in range(num_blocks)}


def _matches(name: str, patterns: list[str]) -> bool:
    lowered = name.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def _get_parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def freeze_non_vector_lora_parameters(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = ".lora_A." in name or ".lora_B." in name


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def list_trainable_parameters(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def vector_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if ".lora_A." in name or ".lora_B." in name
    }


def load_vector_lora_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor], strict: bool = True) -> tuple[list[str], list[str]]:
    own = model.state_dict()
    missing = [name for name in own if (".lora_A." in name or ".lora_B." in name) and name not in state_dict]
    unexpected = [name for name in state_dict if name not in own]
    if strict and (missing or unexpected):
        raise RuntimeError(f"Vector-LoRA state mismatch. Missing={missing[:10]} unexpected={unexpected[:10]}")
    filtered = {name: value for name, value in state_dict.items() if name in own}
    model.load_state_dict(filtered, strict=False)
    return missing, unexpected


def apply_vector_lora(
    model: nn.Module,
    target_patterns: list[str],
    schedule_config: dict[str, Any],
    dropout: float,
    include_modules: list[str] | None = None,
    exclude_modules: list[str] | None = None,
) -> dict[str, Any]:
    if not target_patterns:
        raise ValueError("Vector-LoRA target_patterns must not be empty")
    include_modules = include_modules or []
    exclude_modules = exclude_modules or []
    linear_modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    block_indices = [block for name, _ in linear_modules if (block := infer_block_index(name)) is not None]
    num_blocks = max(block_indices) + 1 if block_indices else 0

    replaced: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for name, module in linear_modules:
        if include_modules and not _matches(name, include_modules):
            continue
        if exclude_modules and _matches(name, exclude_modules):
            skipped.append({"name": name, "reason": "excluded"})
            continue
        if not _matches(name, target_patterns):
            continue
        block_index = infer_block_index(name)
        if block_index is None and not schedule_config.get("default_rank"):
            skipped.append({"name": name, "reason": "no_block_index"})
            warnings.warn(f"Skipping Vector-LoRA target without block index: {name}", stacklevel=2)
            continue
        rank = _scheduled_value(block_index, num_blocks, schedule_config, "rank")
        alpha = _scheduled_value(block_index, num_blocks, schedule_config, "alpha")
        parent, child_name = _get_parent_module(model, name)
        setattr(parent, child_name, VectorLoRALinear(module, rank, alpha, dropout, module_name=name, block_index=block_index))
        replaced.append({"name": name, "rank": rank, "alpha": alpha, "block_index": block_index})

    freeze_non_vector_lora_parameters(model)
    if not replaced:
        raise RuntimeError(
            "Vector-LoRA replacement matched no nn.Linear modules. "
            f"target_patterns={target_patterns}; inspect modules before training."
        )
    trainable = count_trainable_parameters(model)
    total = sum(parameter.numel() for parameter in model.parameters())
    return VectorLoRASummary(
        replaced_modules=replaced,
        skipped_modules=skipped,
        trainable_vector_lora_params=trainable,
        frozen_base_params=total - trainable,
        total_params=total,
        rank_by_module={row["name"]: int(row["rank"]) for row in replaced},
        rank_by_block={str(row["block_index"]): int(row["rank"]) for row in replaced if row["block_index"] is not None},
    ).to_dict()
