# Evaluation Configs

Evaluation runtime profiles live here.

## Current Files

- `default.yaml`: default evaluation profile.
- `hourly.yaml`: reduced-cost evaluation profile.
- `phase1_kaggle_complete.yaml`: Kaggle-oriented evaluation profile.

## Guidelines

- Keep baseline-only and adapted-model evaluations comparable.
- Use the same validation split when comparing configurations.
- Document reported metrics in `results/reports/` or `notebooks/ablations/`.
