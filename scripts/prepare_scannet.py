from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Placeholder ScanNet prep helper.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    args.target.mkdir(parents=True, exist_ok=True)
    print(
        "Expected ScanNet-style directories under the target path. "
        "Copy or link scene folders with color/depth/pose subdirectories."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
