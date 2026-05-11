# Notebooks

This folder contains research notebooks grouped by purpose. Keep notebooks here, not in the repository root.

## Folder Layout

| Folder | Purpose |
|---|---|
| `ablations/` | Final ablation notebooks and the ablation report. |
| `kaggle/` | Kaggle setup, training, evaluation, and complete Phase 1 notebooks. |
| `evaluation/` | Evaluation, comparison, visual inspection, and benchmark notebooks. |
| `experiments/` | Current exploratory training notebooks. |
| `phase1_adapted_longrun/` | Long-run adapted-only Phase 1 experiment and selected checkpoint artifact/pointer. |
| `archive/` | Old, duplicate, or superseded notebooks kept for traceability. |

## Primary Notebook Paths

| Notebook | Purpose |
|---|---|
| `experiments/eaglevision2.ipynb` | Main active training and evaluation notebook. |
| `ablations/kaggle-ablations.ipynb` | Final Kaggle ablation notebook. |
| `ablations/eaglevision_hyperparameter_ablation_local_cuda_v3_unicode_single_progress.ipynb` | Local/CUDA ablation variant. |
| `kaggle/kaggle_phase1_setup_train.ipynb` | Kaggle setup and training. |
| `kaggle/kaggle_phase1_eval_infer.ipynb` | Kaggle evaluation and inference. |
| `kaggle/kaggle_phase1_complete.ipynb` | Complete Kaggle Phase 1 workflow. |
| `evaluation/eval_complete/eval-complete-final.ipynb` | Original final evaluation dashboard notebook. |
| `phase1_adapted_longrun/ablated-full-run.ipynb` | Long-run adapted-only experiment. |

## Reports Connected To Notebooks

| Report | Source |
|---|---|
| `ablations/eaglevision_ablation_report.md` | Final ablation notebooks. |
| `../results/notebooks/README.md` | Complete evaluation report generated from `eval-complete-final.ipynb`. |
| `../results/notebooks/eval-complete-final.ipynb` | Copied notebook placed beside the complete evaluation report. |

## Notebook Policy

- Keep notebooks in the folder that matches their purpose.
- Strip large transient output before committing unless the output is central to the report.
- Put written reports beside the notebook when the report explains that exact notebook's results.
- Use `results/notebooks/README.md` when the README itself should render as an evaluation report on GitHub.
- Move superseded notebooks to `archive/` instead of leaving them in the root.

## Dataset

The Kaggle workflows use:

```text
klein2111/scannet-2d
```

See `docs/kaggle_submission.md` for Kaggle-specific setup.
