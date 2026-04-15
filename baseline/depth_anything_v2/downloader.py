from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

from .checkpoint_registry import CheckpointSpec


def download_checkpoint(spec: CheckpointSpec, target_dir: Path, force: bool = False) -> tuple[Path, bool]:
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / spec.filename

    if destination.exists() and not force:
        return destination, False

    tmp_path = destination.with_suffix(destination.suffix + ".part")
    with urlopen(spec.url) as response, tmp_path.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)

    tmp_path.replace(destination)
    return destination, True
