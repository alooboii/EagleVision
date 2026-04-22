# Repository Guide

Relevant paths:

- `baseline/depth_anything_v2`: vendor baseline, intentionally isolated.
- `src/eaglevision`: Phase 1 source tree.
- `configs/`: train, eval, data, and model YAML files.
- `scripts/`: thin wrappers for common CLI entrypoints.
- `tests/`: small fast tests for geometry, fusion, and pairing assumptions.

Common commands:

```bash
pip install -e .
python -m eaglevision.cli.train --config configs/train/phase1.yaml
python -m eaglevision.cli.eval --config configs/eval/default.yaml
python -m eaglevision.cli.infer_depth --input image.jpg --output outputs/depth.png
```

The repository is organized so later phases can add learned fusion, source-depth prediction regimes, refinement, and generative completion modules without disturbing Phase 1 geometry.
