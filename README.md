# EagleVision

EagleVision is a research codebase for improving monocular depth estimation through lightweight geometric adaptation. The project adapts a pretrained Depth Anything V2-style depth estimator inside a geometry-first RGB-D round-trip novel-view-synthesis workflow for indoor scenes.

The current repository is organized around Phase 1: keep the pretrained depth model mostly frozen, train a lightweight residual adapter, and evaluate whether the adapted model improves direct depth quality and geometric usefulness.

## What This Repository Contains

```text
baseline/       Vendored Depth Anything V2 baseline wrapper
configs/        YAML configs for data, model, training, and evaluation
docs/           Project design notes, experiment notes, and paper assets
notebooks/      Kaggle, ablation, evaluation, and archived research notebooks
outputs/        Local generated outputs only; ignored by git
results/        Small reports and result notebooks
scripts/        Thin command wrappers around package CLIs
src/            EagleVision Python package
tests/          Fast CPU-friendly unit tests
```

## Phase 1 Scope

Phase 1 focuses on:

- loading ScanNet-style paired indoor RGB-D views
- sampling nearby camera pairs with controlled translation and rotation
- rendering target views using explicit RGB-D geometry
- adapting a pretrained depth model with a lightweight residual head
- fusing warped and predicted depth with visibility-aware filling
- training with depth anchoring and round-trip geometric losses
- comparing adapted depth against the frozen pretrained baseline

See [docs/phase1_scope.md](docs/phase1_scope.md), [docs/architecture.md](docs/architecture.md), and [docs/training_plan.md](docs/training_plan.md) for the detailed design.

## Installation

Use Python 3.11 or 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the fast test suite:

```bash
pytest
```

## Main Commands

Train:

```bash
python -m eaglevision.cli.train --config configs/train/phase1.yaml
```

Evaluate:

```bash
python -m eaglevision.cli.eval --config configs/eval/default.yaml
python -m eaglevision.cli.eval --config configs/eval/default.yaml --baseline-only
```

Fast-turn profile:

```bash
python -m eaglevision.cli.train --config configs/train/phase1_hourly.yaml
python -m eaglevision.cli.eval --config configs/eval/hourly.yaml --baseline-only
python -m eaglevision.cli.eval --config configs/eval/hourly.yaml --checkpoint <ckpt.pt>
```

Single-image depth inference:

```bash
python -m eaglevision.cli.infer_depth --input image.jpg --output outputs/depth.png
```

Novel-view render from RGB-D:

```bash
python -m eaglevision.cli.render_novel_view \
  --rgb image.jpg \
  --depth depth.npy \
  --intrinsics intrinsics.txt \
  --transform transform.txt \
  --output outputs/novel_view.png
```

The same commands are exposed as console scripts after installation:

```bash
eaglevision-train
eaglevision-eval
eaglevision-compare
eaglevision-infer-depth
eaglevision-render
eaglevision-demo
```

## Data

Phase 1 expects ScanNet-style scene folders with:

```text
scene_id/
  color/
  depth/
  pose/
```

Default pairing thresholds are conservative:

- max translation: `0.30 m`
- max rotation: `10 deg`
- minimum motion thresholds to avoid near-identical pairs

Data should stay outside git. Use local `data/` or Kaggle inputs, then point configs at the dataset root. See [src/eaglevision/data/README.md](src/eaglevision/data/README.md) and [docs/kaggle_submission.md](docs/kaggle_submission.md).

## Notebooks And Reports

Notebook folders are intentionally separated by purpose:

- [notebooks/ablations](notebooks/ablations): final ablation notebooks and the ablation report
- [notebooks/kaggle](notebooks/kaggle): Kaggle setup, training, evaluation, and complete Phase 1 notebooks
- [notebooks/evaluation](notebooks/evaluation): evaluation and comparison notebooks
- [notebooks/experiments](notebooks/experiments): current exploratory training notebooks
- [notebooks/archive](notebooks/archive): older or duplicate notebooks kept for traceability

The final ablation report lives beside the ablation notebooks at [notebooks/ablations/eaglevision_ablation_report.md](notebooks/ablations/eaglevision_ablation_report.md).

## Baseline

The original baseline integration is isolated under [baseline/depth_anything_v2](baseline/depth_anything_v2). It remains available through:

```bash
python -m baseline.depth_anything_v2 download
python -m baseline.depth_anything_v2 infer --input <image-or-folder> --output-dir outputs/baseline
```

## Repository Rules

- Commit source, configs, docs, tests, and small markdown reports.
- Do not commit raw datasets, generated outputs, local run folders, or large checkpoint artifacts.
- Use Git LFS only for deliberately retained model artifacts.
- Keep notebooks in the folder matching their role.
- Keep `results/` for small reports and result notebooks, not raw experiment dumps.

See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/repository_guide.md](docs/repository_guide.md) for contributor workflow details.

## Documentation Index

- [Repository Guide](docs/repository_guide.md)
- [Architecture](docs/architecture.md)
- [Training Plan](docs/training_plan.md)
- [Experiments](docs/experiments.md)
- [Phase 1 Scope](docs/phase1_scope.md)
- [Kaggle Submission Guide](docs/kaggle_submission.md)
- [Artifact Policy](docs/artifact_policy.md)
- [Reproducibility](docs/reproducibility.md)
