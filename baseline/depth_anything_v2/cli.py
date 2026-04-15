from __future__ import annotations

import argparse
from pathlib import Path

from .checkpoint_registry import VALID_ENCODERS, VALID_PROFILES, default_checkpoints_dir, resolve_download_specs
from .downloader import download_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m baseline.depth_anything_v2",
        description="Depth Anything V2 baseline CLI",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser(
        "download",
        help="Download official Depth Anything V2 checkpoints",
    )
    download_parser.add_argument(
        "--mode",
        choices=["relative", "metric", "all"],
        default="all",
        help="Checkpoint type to download",
    )
    download_parser.add_argument(
        "--encoder",
        action="append",
        choices=VALID_ENCODERS,
        help="Encoder(s) to download. Repeat to specify multiple values.",
    )
    download_parser.add_argument(
        "--profile",
        choices=VALID_PROFILES,
        default="hypersim",
        help="Metric depth profile. Ignored for --mode relative.",
    )
    download_parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=default_checkpoints_dir(),
        help="Destination directory for downloaded checkpoint files",
    )
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even when they already exist",
    )

    infer_parser = subparsers.add_parser(
        "infer",
        help="Run relative or metric depth inference",
    )
    infer_parser.add_argument("--input", type=Path, required=True, help="Input image path or directory")
    infer_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    infer_parser.add_argument(
        "--mode",
        choices=["relative", "metric"],
        default="relative",
        help="Inference mode",
    )
    infer_parser.add_argument(
        "--encoder",
        choices=VALID_ENCODERS,
        default="vitl",
        help="Depth Anything V2 encoder variant",
    )
    infer_parser.add_argument(
        "--profile",
        choices=VALID_PROFILES,
        default="hypersim",
        help="Metric profile (used only with --mode metric)",
    )
    infer_parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Explicit checkpoint path (overrides local cache lookup)",
    )
    infer_parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=default_checkpoints_dir(),
        help="Local checkpoint cache directory",
    )
    infer_parser.add_argument(
        "--input-size",
        type=int,
        default=518,
        help="Inference input size (larger can improve detail at higher cost)",
    )
    infer_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Execution device",
    )

    return parser


def _resolve_encoders(raw_values: list[str] | None) -> list[str]:
    if not raw_values:
        return list(VALID_ENCODERS)

    seen: dict[str, None] = {}
    for value in raw_values:
        seen[value] = None
    return list(seen)


def run_download(args: argparse.Namespace) -> int:
    encoders = _resolve_encoders(args.encoder)
    if args.mode == "relative" and args.profile != "hypersim":
        raise ValueError("--profile is only meaningful for metric/all mode")

    specs = resolve_download_specs(
        mode=args.mode,
        encoders=encoders,
        profile=args.profile,
    )

    for spec in specs:
        path, downloaded = download_checkpoint(spec, target_dir=args.checkpoints_dir, force=args.force)
        status = "downloaded" if downloaded else "already present"
        descriptor = f"{spec.mode}:{spec.encoder}" if spec.profile is None else f"{spec.mode}:{spec.profile}:{spec.encoder}"
        print(f"{descriptor} -> {path} ({status})")

    return 0


def run_infer(args: argparse.Namespace) -> int:
    if args.mode == "relative" and args.profile != "hypersim":
        raise ValueError("--profile can only be set when --mode metric")

    from .inference import run_inference

    run_inference(
        input_path=args.input,
        output_dir=args.output_dir,
        mode=args.mode,
        encoder=args.encoder,
        profile=args.profile,
        input_size=args.input_size,
        device_name=args.device,
        checkpoints_dir=args.checkpoints_dir,
        explicit_checkpoint=args.checkpoint,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "download":
            return run_download(args)
        if args.command == "infer":
            return run_infer(args)
    except Exception as exc:  # noqa: BLE001
        parser.error(str(exc))

    parser.error("Unknown command")
    return 2
