from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


PATH_FIELDS = ("left_rgb", "right_rgb", "depth", "disparity", "mask", "K_left", "K_right", "T_left_to_right")
OPTIONAL_FIELDS = ("depth", "disparity", "mask", "K_left", "K_right", "T_left_to_right", "baseline", "focal_length")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate an endoscopy stereo JSONL manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--require-geometry", "--geometric-training", action="store_true")
    return parser


def _load_rows(manifest: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sample_id(row: dict[str, Any]) -> str:
    return str(row.get("sample_id", f"line_{row.get('_line_number', '?')}"))


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.load()
        return image.size


def validate_manifest(manifest: Path, root: Path | None = None, require_geometry: bool = False) -> dict[str, Any]:
    root = root or manifest.parent
    rows = _load_rows(manifest)
    missing_counts: Counter[str] = Counter()
    errors: list[str] = []
    image_sizes: list[tuple[int, int]] = []
    sequences: dict[str, int] = defaultdict(int)
    field_presence: dict[str, list[str]] = {field: [] for field in OPTIONAL_FIELDS}

    for row in rows:
        sample_id = _sample_id(row)
        sequence_id = str(row.get("sequence_id", ""))
        sequences[sequence_id] += 1

        for required in ("left_rgb", "right_rgb"):
            if not row.get(required):
                missing_counts[required] += 1
                errors.append(f"{sample_id}: missing required field '{required}'")
                continue
            path = _resolve(root, row[required])
            if not path.exists():
                missing_counts[required] += 1
                errors.append(f"{sample_id}: {required} does not exist: {path}")
                continue
            try:
                image_sizes.append(_image_size(path))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{sample_id}: could not load {required}: {path} ({exc})")

        if require_geometry:
            for field in ("baseline", "focal_length"):
                if row.get(field) in (None, ""):
                    missing_counts[field] += 1
                    errors.append(f"{sample_id}: geometric training requires '{field}'")

        for field in OPTIONAL_FIELDS:
            if row.get(field) not in (None, ""):
                field_presence[field].append(sample_id)

        for field in PATH_FIELDS:
            if field in ("left_rgb", "right_rgb") or row.get(field) in (None, ""):
                continue
            path = _resolve(root, row[field])
            if not path.exists():
                missing_counts[field] += 1
                errors.append(f"{sample_id}: {field} does not exist: {path}")

    inconsistent_fields: dict[str, dict[str, Any]] = {}
    sample_count = len(rows)
    for field, present_sample_ids in field_presence.items():
        if 0 < len(present_sample_ids) < sample_count:
            missing_sample_ids = [_sample_id(row) for row in rows if row.get(field) in (None, "")]
            inconsistent_fields[field] = {
                "present": len(present_sample_ids),
                "missing": len(missing_sample_ids),
                "missing_sample_ids": missing_sample_ids,
            }
            if not require_geometry or field not in {"baseline", "focal_length"}:
                missing_counts[field] += len(missing_sample_ids)
            errors.append(
                f"optional field '{field}' is mixed across rows; missing for sample_ids: "
                f"{', '.join(missing_sample_ids)}"
            )

    width_values = [size[0] for size in image_sizes]
    height_values = [size[1] for size in image_sizes]
    image_size_stats = {
        "num_images_loaded": len(image_sizes),
        "unique_sizes": sorted({f"{width}x{height}" for width, height in image_sizes}),
        "min_width": min(width_values) if width_values else None,
        "max_width": max(width_values) if width_values else None,
        "min_height": min(height_values) if height_values else None,
        "max_height": max(height_values) if height_values else None,
    }

    summary = {
        "valid": not errors,
        "num_samples": sample_count,
        "has_depth_gt": len(field_presence["depth"]) == sample_count and sample_count > 0,
        "has_disparity_gt": len(field_presence["disparity"]) == sample_count and sample_count > 0,
        "has_baseline_focal": (
            len(field_presence["baseline"]) == sample_count
            and len(field_presence["focal_length"]) == sample_count
            and sample_count > 0
        ),
        "missing_counts": dict(sorted(missing_counts.items())),
        "image_size_stats": image_size_stats,
        "sequences": dict(sorted(sequences.items())),
        "inconsistent_fields": inconsistent_fields,
        "errors": errors,
    }
    return summary


def main() -> int:
    args = build_argparser().parse_args()
    summary = validate_manifest(args.manifest, root=args.root, require_geometry=args.require_geometry)
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
