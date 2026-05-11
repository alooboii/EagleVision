# Training Configs

Training runtime profiles live here.

## Current Files

- `phase1.yaml`: default Phase 1 training configuration.
- `phase1_hourly.yaml`: reduced-cost profile for faster local iteration.
- `phase1_kaggle_complete.yaml`: Kaggle-oriented Phase 1 training run.

## Guidelines

- Keep long-running experiment configs named clearly.
- Record non-obvious profile choices in `docs/experiments.md` or a nearby result report.
- Keep generated run outputs under `outputs/`, not inside `configs/`.
