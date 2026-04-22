from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from eaglevision.models.depth.depth_anything_wrapper import DepthAnythingWithAdapter
from eaglevision.models.rt_depthnvs import RoundTripDepthNVS
from eaglevision.visualization.save_panels import save_debug_panel


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a synthetic round-trip demo from a saved batch tensor file.")
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    batch = torch.load(args.batch, map_location="cpu")
    model = RoundTripDepthNVS(DepthAnythingWithAdapter(mode="metric", encoder="vitl", profile="hypersim"))
    with torch.no_grad():
        outputs = model(batch)
    save_debug_panel(outputs, args.output)
    print(f"Saved panel to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
