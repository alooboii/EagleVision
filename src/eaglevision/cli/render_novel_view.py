from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from eaglevision.models.nvs.geometric_warp import GeometricWarper


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a geometric novel view from RGB-D.")
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--transform", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    rgb = cv2.cvtColor(cv2.imread(str(args.rgb), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    depth = np.load(args.depth)
    intrinsics = np.loadtxt(args.intrinsics, dtype=np.float32).reshape(3, 3)
    transform = np.loadtxt(args.transform, dtype=np.float32).reshape(4, 4)
    image_t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    depth_t = torch.from_numpy(depth).float().unsqueeze(0)
    k_t = torch.from_numpy(intrinsics).float().unsqueeze(0)
    t_t = torch.from_numpy(transform).float().unsqueeze(0)
    outputs = GeometricWarper()(image_t, depth_t, k_t, k_t, t_t)
    warped = outputs["warped_rgb"][0].permute(1, 2, 0).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor((warped * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
