# Depth Anything V2 Baseline

This package wraps the pretrained Depth Anything V2 baseline used by EagleVision.

## Files

- `cli.py`: baseline command-line entrypoints.
- `downloader.py`: checkpoint download support.
- `inference.py`: image and folder inference logic.
- `modeling.py`: model construction and loading.
- `transforms.py`: input preprocessing.
- `checkpoint_registry.py`: known checkpoint metadata.
- `checkpoints/`: local checkpoint location placeholder.

## Development Rule

Keep this package as close to the upstream baseline behavior as possible. EagleVision-specific model adaptation, training losses, and geometry code belong in `src/eaglevision/`.
