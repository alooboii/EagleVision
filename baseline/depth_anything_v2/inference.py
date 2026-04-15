from __future__ import annotations

from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

from .checkpoint_registry import checkpoint_spec
from .io_utils import collect_images, relative_output_stem, save_outputs
from .modeling import create_model
from .transforms import preprocess_bgr_image


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")

    return torch.device(requested)


def resolve_checkpoint_path(
    mode: str,
    encoder: str,
    profile: str,
    checkpoints_dir: Path,
    explicit_checkpoint: Path | None,
) -> Path:
    if explicit_checkpoint is not None:
        if not explicit_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint file does not exist: {explicit_checkpoint}")
        return explicit_checkpoint

    spec = checkpoint_spec(mode=mode, encoder=encoder, profile=profile)
    checkpoint_path = checkpoints_dir / spec.filename
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Run the download command first or pass --checkpoint."
        )
    return checkpoint_path


def _extract_state_dict(loaded_obj: object) -> dict[str, torch.Tensor]:
    if isinstance(loaded_obj, dict):
        for key in ("state_dict", "model", "module"):
            candidate = loaded_obj.get(key)
            if isinstance(candidate, dict):
                return _strip_module_prefix(candidate)
        if all(isinstance(v, torch.Tensor) for v in loaded_obj.values()):
            return _strip_module_prefix(loaded_obj)

    raise ValueError("Unsupported checkpoint format; expected a state dict-like object")


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict

    if all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def load_model(
    mode: str,
    encoder: str,
    profile: str,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    model = create_model(mode=mode, encoder=encoder, profile=profile)
    checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint_obj)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        missing = ", ".join(missing_keys[:5])
        unexpected = ", ".join(unexpected_keys[:5])
        raise RuntimeError(
            "Checkpoint is incompatible with this implementation. "
            f"Missing keys: [{missing}] Unexpected keys: [{unexpected}]"
        )

    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def infer_single_image(
    model: torch.nn.Module,
    image_bgr,
    input_size: int,
    device: torch.device,
):
    prepared = preprocess_bgr_image(image_bgr, input_size=input_size, device=device)
    depth = model(prepared.tensor)
    depth = F.interpolate(
        depth[:, None],
        prepared.original_size,
        mode="bilinear",
        align_corners=True,
    )[0, 0]
    return depth.cpu().numpy()


def run_inference(
    input_path: Path,
    output_dir: Path,
    mode: str,
    encoder: str,
    profile: str,
    input_size: int,
    device_name: str,
    checkpoints_dir: Path,
    explicit_checkpoint: Path | None,
) -> None:
    collection = collect_images(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(device_name)
    checkpoint_path = resolve_checkpoint_path(
        mode=mode,
        encoder=encoder,
        profile=profile,
        checkpoints_dir=checkpoints_dir,
        explicit_checkpoint=explicit_checkpoint,
    )

    model = load_model(
        mode=mode,
        encoder=encoder,
        profile=profile,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    total = len(collection.images)
    for index, image_path in enumerate(collection.images, start=1):
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        depth = infer_single_image(model, image_bgr=image_bgr, input_size=input_size, device=device)
        output_stem = relative_output_stem(image_path=image_path, input_path=input_path)
        output_base = output_dir / output_stem
        npy_path, png_path = save_outputs(output_base=output_base, depth=depth)

        print(f"[{index}/{total}] {image_path} -> {npy_path} and {png_path}")
