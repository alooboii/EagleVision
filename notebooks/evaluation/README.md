# Evaluation Notebooks

This folder contains notebooks for model comparison, metric inspection, qualitative evaluation, and benchmark-style checks.

## Files

| File | Purpose |
|---|---|
| `eval-complete.ipynb` | Complete evaluation workflow notebook. |
| `eval-suite.ipynb` | Evaluation suite notebook. |
| `eval_complete/eval-complete-final.ipynb` | Original final evaluation dashboard notebook. |
| `multi_model_multi_dataset_depth_benchmark.ipynb` | Multi-model, multi-dataset depth benchmark. |
| `quick-check-multi-dataset.ipynb` | Quick multi-dataset sanity check. |
| `quick_adapted_depth_preview.ipynb` | Qualitative adapted depth preview. |

## Rendered Evaluation Report

The GitHub-rendered complete evaluation report is in:

```text
results/notebooks/README.md
```

The copied notebook beside that report is:

```text
results/notebooks/eval-complete-final.ipynb
```

Keep the original notebook here for evaluation workflow history, and use the copied notebook in `results/notebooks/` when reviewing the final reported result.

## Guidelines

- Keep comparisons explicit about dataset split, checkpoint, and config.
- Move final written summaries to `results/` unless they belong beside a specific notebook.
- Keep heavy generated images and arrays out of git unless they are curated report artifacts.
