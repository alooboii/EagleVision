# Notebooks

This folder contains research notebooks grouped by purpose. Keep notebooks here, not in the repository root.

## Folder Layout

- `ablations/`: final ablation notebooks and the ablation report.
- `kaggle/`: Kaggle setup, training, evaluation, and complete Phase 1 notebooks.
- `evaluation/`: evaluation, comparison, and visual inspection notebooks.
- `experiments/`: current exploratory training notebooks.
- `phase1_adapted_longrun/`: long-run adapted-only Phase 1 experiment.
- `archive/`: old, duplicate, or superseded notebooks kept for traceability.

## Primary Notebook Paths

- Main training notebook: `experiments/eaglevision2.ipynb`
- Final ablation notebook: `ablations/kaggle-ablations.ipynb`
- Local/CUDA ablation notebook: `ablations/eaglevision_hyperparameter_ablation_local_cuda_v3_unicode_single_progress.ipynb`
- Final ablation report: `ablations/eaglevision_ablation_report.md`
- Kaggle setup/training: `kaggle/kaggle_phase1_setup_train.ipynb`
- Kaggle evaluation/inference: `kaggle/kaggle_phase1_eval_infer.ipynb`
- Complete Kaggle Phase 1 workflow: `kaggle/kaggle_phase1_complete.ipynb`

## Notebook Policy

- Keep notebooks in the folder that matches their purpose.
- Strip large transient output before committing unless the output is central to the report.
- Put written reports beside the notebook when the report explains that exact notebook's results.
- Move superseded notebooks to `archive/` instead of leaving them in the root.

## Dataset

The Kaggle workflows use:

```text
klein2111/scannet-2d
```

See `docs/kaggle_submission.md` for Kaggle-specific setup.
