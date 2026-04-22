from __future__ import annotations

import argparse
from pathlib import Path

from eaglevision.utils.io import load_yaml


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare baseline and adapted evaluation logs.")
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--adapted-metrics", type=Path, required=True)
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    baseline = load_yaml(args.baseline_metrics)
    adapted = load_yaml(args.adapted_metrics)
    keys = sorted(set(baseline) & set(adapted))
    for key in keys:
        delta = adapted[key] - baseline[key]
        print(f"{key}: baseline={baseline[key]:.6f} adapted={adapted[key]:.6f} delta={delta:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
