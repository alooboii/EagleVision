# CLI

Command-line entrypoints for training, evaluation, inference, rendering, comparison, and demos.

## Entrypoints

- `train.py`: train an EagleVision configuration.
- `eval.py`: evaluate baseline or adapted models.
- `compare_models.py`: compare baseline and adapted outputs.
- `infer_depth.py`: run single-image depth inference.
- `render_novel_view.py`: render from RGB-D, intrinsics, and camera transform.
- `demo_roundtrip.py`: run a round-trip demo batch.

These modules are exposed through console scripts in `pyproject.toml`.
