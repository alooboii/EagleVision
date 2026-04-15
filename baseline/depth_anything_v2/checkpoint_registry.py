from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

VALID_ENCODERS = ("vits", "vitb", "vitl")
VALID_PROFILES = ("hypersim", "vkitti")


@dataclass(frozen=True)
class CheckpointSpec:
    mode: str
    encoder: str
    profile: str | None
    filename: str
    url: str


def default_checkpoints_dir() -> Path:
    return Path(__file__).resolve().parent / "checkpoints"


def checkpoint_spec(mode: str, encoder: str, profile: str = "hypersim") -> CheckpointSpec:
    if encoder not in VALID_ENCODERS:
        raise ValueError(f"Unsupported encoder '{encoder}'. Expected one of: {', '.join(VALID_ENCODERS)}")

    if mode == "relative":
        return _relative_spec(encoder)
    if mode == "metric":
        if profile not in VALID_PROFILES:
            raise ValueError(f"Unsupported profile '{profile}'. Expected one of: {', '.join(VALID_PROFILES)}")
        return _metric_spec(encoder, profile)

    raise ValueError("mode must be one of: relative, metric")


def resolve_download_specs(mode: str, encoders: Iterable[str], profile: str) -> list[CheckpointSpec]:
    encoder_list = list(encoders)
    if mode == "relative":
        return [checkpoint_spec("relative", encoder, profile) for encoder in encoder_list]
    if mode == "metric":
        return [checkpoint_spec("metric", encoder, profile) for encoder in encoder_list]
    if mode == "all":
        relative = [checkpoint_spec("relative", encoder, profile) for encoder in encoder_list]
        metric = [checkpoint_spec("metric", encoder, profile) for encoder in encoder_list]
        return relative + metric

    raise ValueError("mode must be one of: relative, metric, all")


def _relative_spec(encoder: str) -> CheckpointSpec:
    repo_by_encoder = {
        "vits": "Depth-Anything-V2-Small",
        "vitb": "Depth-Anything-V2-Base",
        "vitl": "Depth-Anything-V2-Large",
    }
    filename = f"depth_anything_v2_{encoder}.pth"
    url = (
        "https://huggingface.co/depth-anything/"
        f"{repo_by_encoder[encoder]}/resolve/main/{filename}?download=true"
    )
    return CheckpointSpec(mode="relative", encoder=encoder, profile=None, filename=filename, url=url)


def _metric_spec(encoder: str, profile: str) -> CheckpointSpec:
    repo_by_profile_encoder = {
        "hypersim": {
            "vits": "Depth-Anything-V2-Metric-Hypersim-Small",
            "vitb": "Depth-Anything-V2-Metric-Hypersim-Base",
            "vitl": "Depth-Anything-V2-Metric-Hypersim-Large",
        },
        "vkitti": {
            "vits": "Depth-Anything-V2-Metric-VKITTI-Small",
            "vitb": "Depth-Anything-V2-Metric-VKITTI-Base",
            "vitl": "Depth-Anything-V2-Metric-VKITTI-Large",
        },
    }

    filename = f"depth_anything_v2_metric_{profile}_{encoder}.pth"
    url = (
        "https://huggingface.co/depth-anything/"
        f"{repo_by_profile_encoder[profile][encoder]}/resolve/main/{filename}?download=true"
    )
    return CheckpointSpec(mode="metric", encoder=encoder, profile=profile, filename=filename, url=url)
