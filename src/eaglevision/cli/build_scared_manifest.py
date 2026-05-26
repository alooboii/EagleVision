from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from eaglevision.utils.io import ensure_dir


OPTIONAL_COLUMNS = {
    "depth",
    "disparity",
    "mask",
    "K_left",
    "K_right",
    "T_left_to_right",
    "baseline",
    "focal_length",
    "split",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build SCARED-style stereo JSONL manifests.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mapping-csv", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _relative_or_str(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _infer_sequence(path: Path) -> str:
    parts = path.parts
    for part in parts:
        lowered = part.lower()
        if any(token in lowered for token in ("seq", "sequence", "dataset", "keyframe")):
            return part
    return path.parent.name


def _infer_frame(path: Path) -> str:
    match = re.search(r"(\d+)(?!.*\d)", path.stem)
    return match.group(1) if match else path.stem


def _pair_key(path: Path) -> tuple[str, str] | None:
    text = path.as_posix()
    replacements = [
        ("/left/", "/<side>/"),
        ("/right/", "/<side>/"),
        ("/image_02/", "/<side>/"),
        ("/image_03/", "/<side>/"),
        ("left", "<side>"),
        ("right", "<side>"),
        ("_l.", "_<side>."),
        ("_r.", "_<side>."),
        ("_L.", "_<side>."),
        ("_R.", "_<side>."),
    ]
    side = None
    key = text
    lowered = f"/{text.lower()}/"
    if "/left/" in lowered or "left" in path.stem.lower() or "_l" in path.stem.lower():
        side = "left"
    if "/right/" in lowered or "right" in path.stem.lower() or "_r" in path.stem.lower():
        side = "right" if side is None else side
    if side is None:
        return None
    for src, dst in replacements:
        key = key.replace(src, dst)
    return key, side


def discover_pairs(root: Path) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Path]] = defaultdict(dict)
    for path in root.rglob("*"):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        paired = _pair_key(path)
        if paired is None:
            continue
        key, side = paired
        grouped[key][side] = path

    rows: list[dict[str, Any]] = []
    for sides in grouped.values():
        if "left" not in sides or "right" not in sides:
            continue
        left = sides["left"]
        right = sides["right"]
        sequence_id = _infer_sequence(left.relative_to(root))
        frame_id = _infer_frame(left)
        rows.append(
            {
                "sample_id": f"{sequence_id}_{frame_id}",
                "sequence_id": sequence_id,
                "frame_id": frame_id,
                "left_rgb": _relative_or_str(left, root),
                "right_rgb": _relative_or_str(right, root),
            }
        )
    rows.sort(key=lambda row: (row["sequence_id"], row["frame_id"], row["sample_id"]))
    return rows


def read_mapping_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"left_rgb", "right_rgb"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Mapping CSV is missing required columns: {sorted(missing)}")
        for index, raw in enumerate(reader):
            row = {key: value for key, value in raw.items() if value not in (None, "")}
            row.setdefault("sample_id", f"sample_{index:06d}")
            row.setdefault("sequence_id", "sequence_unknown")
            row.setdefault("frame_id", str(index))
            rows.append(row)
    return rows


def split_rows(rows: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    if any(row.get("split") for row in rows):
        splits = {"train": [], "val": [], "test": []}
        for row in rows:
            split = str(row.get("split", "train")).lower()
            if split not in splits:
                raise ValueError(f"Unsupported split '{split}' for sample {row.get('sample_id')}")
            clean = dict(row)
            clean.pop("split", None)
            splits[split].append(clean)
        return splits

    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sequence[str(row.get("sequence_id", "sequence_unknown"))].append(row)

    sequences = sorted(by_sequence)
    rng = random.Random(seed)
    rng.shuffle(sequences)
    n = len(sequences)
    if n >= 3:
        n_train = max(1, int(round(0.7 * n)))
        n_val = max(1, int(round(0.15 * n)))
        train_seq = set(sequences[:n_train])
        val_seq = set(sequences[n_train : n_train + n_val])
        test_seq = set(sequences[n_train + n_val :]) or val_seq
    elif n == 2:
        train_seq = {sequences[0]}
        val_seq = {sequences[1]}
        test_seq = {sequences[1]}
    else:
        train_seq = val_seq = test_seq = set(sequences)

    return {
        "train": [row for seq in train_seq for row in by_sequence[seq]],
        "val": [row for seq in val_seq for row in by_sequence[seq]],
        "test": [row for seq in test_seq for row in by_sequence[seq]],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    args = build_argparser().parse_args()
    root = args.root
    out_dir = ensure_dir(args.out_dir)

    if args.mapping_csv:
        rows = read_mapping_csv(args.mapping_csv)
        mode = "mapping_csv"
    else:
        rows = discover_pairs(root)
        mode = "auto_discovery"
        if not rows:
            raise RuntimeError(
                "Could not confidently discover left/right image pairs. "
                "Provide --mapping-csv with columns: sample_id,sequence_id,frame_id,left_rgb,right_rgb,"
                "depth,disparity,mask,K_left,K_right,T_left_to_right,baseline,focal_length,split"
            )

    splits = split_rows(rows, args.seed)
    for split, split_rows_ in splits.items():
        write_jsonl(out_dir / f"{split}.jsonl", split_rows_)

    summary = {
        "root": str(root),
        "mode": mode,
        "total_samples": len(rows),
        "splits": {split: len(split_rows_) for split, split_rows_ in splits.items()},
        "sequences": sorted({str(row.get("sequence_id", "")) for row in rows}),
    }
    (out_dir / "manifest_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md = [
        "# SCARED Manifest Summary",
        "",
        f"- Mode: `{mode}`",
        f"- Root: `{root}`",
        f"- Total samples: {len(rows)}",
        f"- Train/val/test: {summary['splits'].get('train', 0)} / {summary['splits'].get('val', 0)} / {summary['splits'].get('test', 0)}",
        f"- Sequences: {len(summary['sequences'])}",
    ]
    (out_dir / "manifest_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote manifests to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
