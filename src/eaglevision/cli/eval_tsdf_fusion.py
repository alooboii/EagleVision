from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

from eaglevision.utils.io import ensure_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TSDF fusion scaffold for 3D-from-X evaluation.")
    parser.add_argument("--frame-manifest", type=Path, required=True)
    parser.add_argument("--pred-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames-per-scene", type=int, default=20)
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    has_open3d = importlib.util.find_spec("open3d") is not None
    ensure_dir(args.out.parent)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "scene_id", "num_frames", "num_vertices", "num_triangles", "status"])
        writer.writeheader()
        writer.writerow({"method": "", "scene_id": "", "num_frames": 0, "num_vertices": "", "num_triangles": "", "status": "open3d_not_installed" if not has_open3d else "not_implemented"})
    print(f"Wrote TSDF scaffold status to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
