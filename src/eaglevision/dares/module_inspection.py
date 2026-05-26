from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from torch import nn

from eaglevision.models.depth.frozen_depth_prior import build_frozen_depth_prior
from eaglevision.utils.io import load_yaml


def inspect_dares_modules(config_path: Path, contains: list[str]) -> list[dict[str, Any]]:
    config = load_yaml(config_path)
    prior = build_frozen_depth_prior(config.get("base_model", {}))
    tokens = [token.lower() for token in contains]
    rows: list[dict[str, Any]] = []
    for name, module in prior.backbone.named_modules():
        class_name = module.__class__.__name__
        haystack = f"{name} {class_name}".lower()
        if tokens and not any(token in haystack for token in tokens):
            continue
        block_match = re.search(r"(?:^|\\.)(?:blocks|block)\\.(\\d+)(?:\\.|$)", name)
        is_linear = isinstance(module, nn.Linear)
        rows.append(
            {
                "name": name,
                "class_name": class_name,
                "parameter_count": sum(parameter.numel() for parameter in module.parameters(recurse=False)),
                "is_linear": is_linear,
                "in_features": module.in_features if is_linear else None,
                "out_features": module.out_features if is_linear else None,
                "block_index": int(block_match.group(1)) if block_match else None,
                "recommended_for_vector_lora": is_linear and any(token in name.lower() for token in ("qkv", "proj", "fc1", "fc2")),
            }
        )
    return rows


def recommended_targets(rows: list[dict[str, Any]]) -> list[str]:
    targets = sorted({str(row["name"]) for row in rows if row.get("recommended_for_vector_lora")})
    return targets
