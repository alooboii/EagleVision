from __future__ import annotations

import argparse
import json
from pathlib import Path

from eaglevision.dares.trainer import run_dry_run, train_dares
from eaglevision.utils.io import load_yaml


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DARES Vector-LoRA on SCARED endoscopy stereo data.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=Path)
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    config = load_yaml(args.config)
    if args.dry_run:
        metrics = run_dry_run(config, args.train_manifest, args.val_manifest)
        print(json.dumps({"dry_run": "ok", **metrics}, indent=2))
        return 0
    train_dares(config, args.train_manifest, args.val_manifest, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
