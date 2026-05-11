# Reproducibility

This document records the expected path from a clean clone to tests, training, evaluation, ablation review, and final reports.

## Environment

Use Python 3.11 or 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Fast Verification

```bash
pytest
```

The fast tests cover geometry utilities, intrinsics scaling, pair sampling, fusion behavior, visibility masks, and basic model shape assumptions.

## Data Assumptions

Phase 1 expects ScanNet-style scenes:

```text
scene_id/
  color/
  depth/
  pose/
```

Point configs at the dataset root. Do not commit the dataset.

## Training

Default local profile:

```bash
python -m eaglevision.cli.train --config configs/train/phase1.yaml
```

Faster CUDA profile:

```bash
python -m eaglevision.cli.train --config configs/train/phase1_hourly.yaml
```

Kaggle-oriented profile:

```bash
python -m eaglevision.cli.train --config configs/train/phase1_kaggle_complete.yaml
```

## Evaluation

Baseline-only:

```bash
python -m eaglevision.cli.eval --config configs/eval/default.yaml --baseline-only
```

Adapted checkpoint:

```bash
python -m eaglevision.cli.eval --config configs/eval/default.yaml --checkpoint <ckpt.pt>
```

Reduced-cost CUDA evaluation:

```bash
python -m eaglevision.cli.eval --config configs/eval/hourly.yaml --baseline-only
python -m eaglevision.cli.eval --config configs/eval/hourly.yaml --checkpoint <ckpt.pt>
```

## Ablations

The final ablation notebook and report are in `notebooks/ablations/`.

Use:

- `notebooks/ablations/kaggle-ablations.ipynb`
- `notebooks/ablations/eaglevision_hyperparameter_ablation_local_cuda_v3_unicode_single_progress.ipynb`
- `notebooks/ablations/eaglevision_ablation_report.md`

## Complete Evaluation Report

The complete evaluation report is rendered by GitHub from:

```text
results/notebooks/README.md
```

The copied source notebook for that report is:

```text
results/notebooks/eval-complete-final.ipynb
```

The original evaluation notebook path is:

```text
notebooks/evaluation/eval_complete/eval-complete-final.ipynb
```

## Kaggle

Kaggle-specific notebooks are in `notebooks/kaggle/`.

Expected public input dataset:

```text
klein2111/scannet-2d
```

See `docs/kaggle_submission.md` for Kaggle-specific setup.
