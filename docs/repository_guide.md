# Repository Guide

This guide explains how the current EagleVision repository is organized and where new work should go.

## Top-Level Layout

| Path | Purpose |
|---|---|
| `baseline/` | Preserved Depth Anything V2-style baseline integration. Keep baseline-specific behavior isolated here. |
| `configs/` | YAML runtime profiles for data, model, training, and evaluation. |
| `docs/` | Project architecture, scope, reproducibility, artifact, training, and experiment documentation. |
| `notebooks/` | Research notebooks grouped by role: ablations, Kaggle, evaluation, experiments, archive, and long-run work. |
| `outputs/` | Local generated outputs only. Git tracks only docs/placeholders here. |
| `results/` | Rendered reports and result-inspection notebooks. |
| `scripts/` | Thin wrappers around package CLIs. |
| `src/eaglevision/` | Main installable EagleVision package. |
| `tests/` | Fast CPU-friendly unit tests. |
| `.github/` | GitHub Actions, PR template, and issue templates. |

## Source Layout

| Path | Purpose |
|---|---|
| `src/eaglevision/cli/` | User-facing train/eval/infer/render/demo entrypoints. |
| `src/eaglevision/data/` | ScanNet-style dataset loading, pair sampling, transforms, and collation. |
| `src/eaglevision/engine/` | Trainer, evaluator, checkpointing, and metric logging. |
| `src/eaglevision/losses/` | Phase 1 loss functions and total objective composition. |
| `src/eaglevision/metrics/` | Depth, image, and geometric consistency metrics. |
| `src/eaglevision/models/` | Adapter, depth wrapper, fusion, novel-view geometry, refinement, and round-trip model. |
| `src/eaglevision/utils/` | Geometry, intrinsics, I/O, masks, seeding, and visualization helpers. |
| `src/eaglevision/visualization/` | Debug/result panel generation. |

## Notebook Layout

| Path | Purpose |
|---|---|
| `notebooks/ablations/` | Final ablation notebooks and `eaglevision_ablation_report.md`. |
| `notebooks/kaggle/` | Kaggle setup, training, evaluation, and complete workflow notebooks. |
| `notebooks/evaluation/` | Evaluation and comparison notebooks. |
| `notebooks/experiments/` | Active exploratory training notebooks. |
| `notebooks/phase1_adapted_longrun/` | Long-run adapted-only Phase 1 experiment and selected checkpoint pointer/artifact. |
| `notebooks/archive/` | Older or duplicate notebooks kept for traceability, not primary entrypoints. |

## Reports

| Path | Purpose |
|---|---|
| `results/notebooks/README.md` | Complete evaluation report generated from `eval-complete-final.ipynb`. |
| `results/notebooks/eval-complete-final.ipynb` | Copied source notebook for the complete evaluation report. |
| `notebooks/ablations/eaglevision_ablation_report.md` | Final ablation report and configuration justification. |
| `results/reports/100ep_vs_Base.md` | 100-epoch adapted run compared with the baseline. |
| `results/reports/post_ablation_training_100ep.md` | Post-ablation 100-epoch training summary. |

## Common Commands

Install:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Train:

```bash
python -m eaglevision.cli.train --config configs/train/phase1.yaml
```

Evaluate:

```bash
python -m eaglevision.cli.eval --config configs/eval/default.yaml
python -m eaglevision.cli.eval --config configs/eval/default.yaml --baseline-only
```

Run a fast-turn CUDA profile:

```bash
python -m eaglevision.cli.train --config configs/train/phase1_hourly.yaml
python -m eaglevision.cli.eval --config configs/eval/hourly.yaml --baseline-only
```

## Where New Work Should Go

- New reusable training or evaluation code: `src/eaglevision/`.
- New CLI behavior: `src/eaglevision/cli/`.
- New runtime settings: `configs/`.
- New fast tests: `tests/`.
- New experiment reports: `results/reports/` unless they directly explain a notebook in another folder.
- New ablation report: `notebooks/ablations/` beside the ablation notebook.
- New generated outputs: ignored `outputs/` or external artifact storage.

## Artifact Rules

Commit source, configs, docs, tests, and small markdown reports.

Do not commit:

- raw datasets
- local generated `outputs/`
- local run folders
- credentials
- large checkpoints unless deliberately tracked with Git LFS

Use `docs/artifact_policy.md` for the full policy.
