from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch import nn

from eaglevision.models.depth.frozen_depth_prior import build_frozen_depth_prior
from eaglevision.utils.io import load_yaml


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect Depth Anything V2 module names for LoRA targeting.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--contains", nargs="+", default=[])
    parser.add_argument("--out", type=Path)
    return parser


def _param_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters(recurse=False))


def inspect_modules(config_path: Path, contains: list[str]) -> list[dict[str, object]]:
    config = load_yaml(config_path)
    prior = build_frozen_depth_prior(config.get("base_model", {}))
    tokens = [token.lower() for token in contains]
    rows: list[dict[str, object]] = []
    for name, module in prior.backbone.named_modules():
        class_name = module.__class__.__name__
        haystack = f"{name} {class_name}".lower()
        if tokens and not any(token in haystack for token in tokens):
            continue
        rows.append(
            {
                "name": name,
                "class_name": class_name,
                "parameter_count": _param_count(module),
                "is_linear_lora_candidate": isinstance(module, nn.Linear),
            }
        )
    return rows


def main() -> int:
    args = build_argparser().parse_args()
    rows = inspect_modules(args.config, args.contains)
    for row in rows:
        marker = " [LoRA candidate]" if row["is_linear_lora_candidate"] else ""
        print(f"{row['name']}: {row['class_name']} params={row['parameter_count']}{marker}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
