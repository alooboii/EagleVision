from __future__ import annotations

import argparse
import csv
from pathlib import Path

from eaglevision.utils.io import ensure_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Depth-to-mesh scaffold for 3D-from-X evaluation.")
    parser.add_argument("--pred-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    ensure_dir(args.out.parent)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "frame_id", "scene_id", "num_vertices", "num_faces", "status"])
        writer.writeheader()
        writer.writerow({"method": "", "frame_id": "", "scene_id": "", "num_vertices": "", "num_faces": "", "status": "not_implemented"})
    print(f"Wrote depth-to-mesh scaffold status to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
