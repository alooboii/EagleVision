# Experiments

This document summarizes the active experiment families represented in the current repository.

## Current Experiment Families

| Experiment | Primary Artifacts | Purpose |
|---|---|---|
| Baseline depth evaluation | `configs/eval/default.yaml`, `configs/eval/hourly.yaml`, `python -m eaglevision.cli.eval --baseline-only` | Measure frozen Depth Anything V2 behavior under the same pair sampling and metrics. |
| Adapted depth evaluation | `python -m eaglevision.cli.eval --checkpoint <ckpt.pt>` | Evaluate a trained EagleVision adapter against the baseline. |
| Phase 1 round-trip training | `configs/train/phase1.yaml`, `configs/train/phase1_hourly.yaml` | Train the residual adapter through geometry-first round-trip losses. |
| Kaggle Phase 1 workflow | `notebooks/kaggle/` | Run setup, training, and evaluation inside Kaggle runtimes. |
| Final ablation study | `notebooks/ablations/kaggle-ablations.ipynb` and `notebooks/ablations/eaglevision_ablation_report.md` | Compare loss weights, learning rate, adapter capacity, RGB losses, and motion thresholds. |
| Complete evaluation dashboard | `results/notebooks/README.md` and `results/notebooks/eval-complete-final.ipynb` | Report final 100-epoch adapted-vs-baseline metrics, paired tests, motion buckets, scale diagnostics, and runtime. |
| 100-epoch long run | `notebooks/phase1_adapted_longrun/` and `results/reports/100ep_vs_Base.md` | Inspect the longer adapted-only Phase 1 run. |

## Recommended Comparisons

1. Frozen DAV2 baseline versus adapted model on target-depth metrics.
2. Frozen DAV2 baseline versus adapted model inside the same round-trip evaluation loop.
3. Adapted 100-epoch checkpoint versus baseline across motion buckets.
4. Ablation configurations versus default weights and frozen baseline.
5. Raw depth metrics versus median-scaled metrics to separate scale quality from local structure.
6. Full-image image metrics versus valid-overlap metrics to separate rendering quality from projection coverage.

## Current Findings

The ablation report supports the project direction: lightweight adapter training improves over the frozen baseline in the reduced-budget validation setup, with `lr_5e-5` identified as the best overall ablation configuration.

The complete evaluation report is more nuanced. The adapted 100-epoch model improves depth/geometric metrics such as AbsRel, RMSE, Delta1, valid-region SSIM, and reprojection depth L1, but regresses some full-image reconstruction and coverage metrics because holes/invalid projected regions remain a limitation.

The strongest current claim is:

> EagleVision improves depth/geometric accuracy over the frozen pretrained baseline with minimal runtime overhead, while full-frame novel-view reconstruction quality remains limited by visibility coverage, holes, and geometry setup.

## Reporting Rules

- Put ablation-specific written reports beside ablation notebooks.
- Put complete evaluation reports under `results/notebooks/` when the folder README is meant to render on GitHub.
- Put small standalone experiment summaries under `results/reports/`.
- Keep raw run outputs and bulky artifacts out of git.
