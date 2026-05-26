from __future__ import annotations

import argparse
import json
from pathlib import Path

from eaglevision.dares.module_inspection import inspect_dares_modules, recommended_targets


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect DA2 modules for DARES Vector-LoRA insertion.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--contains", nargs="+", default=["attn", "qkv", "proj", "mlp", "linear"])
    parser.add_argument("--out", type=Path)
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    rows = inspect_dares_modules(args.config, args.contains)
    for row in rows:
        marker = " [recommended Vector-LoRA]" if row["recommended_for_vector_lora"] else ""
        print(
            f"{row['name']}: {row['class_name']} params={row['parameter_count']} "
            f"linear={row['is_linear']} in={row['in_features']} out={row['out_features']} block={row['block_index']}{marker}"
        )
    targets = recommended_targets(rows)
    print("\nRecommended DARES target module names:")
    for target in targets:
        print(f"  - {target}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"modules": rows, "recommended_targets": targets}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
