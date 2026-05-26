from __future__ import annotations

import argparse
from pathlib import Path

from eaglevision.dares.evaluator import evaluate_dares
from eaglevision.utils.io import load_yaml


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate DARES Vector-LoRA on SCARED endoscopy data.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    evaluate_dares(load_yaml(args.config), args.manifest, args.checkpoint, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
