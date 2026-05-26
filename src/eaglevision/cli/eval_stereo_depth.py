from __future__ import annotations

import argparse
import csv
from pathlib import Path

from eaglevision.utils.io import ensure_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optional stereo depth scaffold for 3D-from-X suite.")
    parser.add_argument("--stereo-manifest", type=Path)
    parser.add_argument("--pred-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    ensure_dir(args.out.parent)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "sample_id", "disp_absrel", "disp_rmse", "photo_l1", "status"])
        writer.writeheader()
        status = "missing_stereo_manifest" if args.stereo_manifest is None or not args.stereo_manifest.exists() else "not_implemented"
        writer.writerow({"method": "", "sample_id": "", "disp_absrel": "", "disp_rmse": "", "photo_l1": "", "status": status})
    print(f"Wrote stereo scaffold status to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
