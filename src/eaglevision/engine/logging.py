from __future__ import annotations

from pathlib import Path
import json

from eaglevision.utils.io import ensure_dir


class JsonlLogger:
    """Minimal JSONL metrics logger."""

    def __init__(self, path: Path) -> None:
        ensure_dir(path.parent)
        self.path = path

    def log(self, payload: dict) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
