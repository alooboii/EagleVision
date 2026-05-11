# Evaluation Notebooks

This folder contains notebooks for model comparison, metric inspection, and qualitative evaluation.

## Files

- `eval-complete.ipynb`: complete evaluation workflow.
- `eval-suite.ipynb`: evaluation suite notebook.
- `eval_complete/`: final evaluation notebook folder.
- `multi_model_multi_dataset_depth_benchmark.ipynb`: multi-model/multi-dataset comparison.
- `quick-check-multi-dataset.ipynb`: quick multi-dataset sanity check.
- `quick_adapted_depth_preview.ipynb`: adapted depth preview notebook.

## Guidelines

- Keep comparisons explicit about dataset split, checkpoint, and config.
- Move final written summaries to `results/reports/`.
- Keep heavy generated images and arrays out of git unless they are curated paper artifacts.
