# Reproducibility

This document records the expected path from a clean clone to tests, training, evaluation, and notebook reports.

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

These tests cover geometry utilities, intrinsics scaling, pair sampling, fusion behavior, and basic model shape assumptions.

## Training

```bash
python -m eaglevision.cli.train --config configs/train/phase1.yaml
```

For faster iteration:

```bash
python -m eaglevision.cli.train --config configs/train/phase1_hourly.yaml
```

## Evaluation

```bash
python -m eaglevision.cli.eval --config configs/eval/default.yaml --baseline-only
python -m eaglevision.cli.eval --config configs/eval/default.yaml --checkpoint <ckpt.pt>
```

## Ablations

The final ablation notebook and report are in `notebooks/ablations/`.

Use `notebooks/ablations/kaggle-ablations.ipynb` for the Kaggle ablation workflow, then read `notebooks/ablations/eaglevision_ablation_report.md` for the summarized findings.

## Kaggle

Kaggle-specific notebooks are in `notebooks/kaggle/`. The expected public input dataset is documented in `docs/kaggle_submission.md`.
