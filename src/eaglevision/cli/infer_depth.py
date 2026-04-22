from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch

from eaglevision.models.depth.depth_anything_wrapper import DepthAnythingWithAdapter
from eaglevision.utils.visualization import depth_to_colormap


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run adapted depth inference on a single image.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--mode", default="metric", choices=["metric", "relative"])
    parser.add_argument("--encoder", default="vitl", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--profile", default="hypersim", choices=["hypersim", "vkitti"])
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    bgr = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    model = DepthAnythingWithAdapter(
        mode=args.mode,
        encoder=args.encoder,
        profile=args.profile,
        checkpoint_path=args.checkpoint,
    )
    with torch.no_grad():
        depth = model(tensor)["adapted_depth"][0]
    preview = depth_to_colormap(depth)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
