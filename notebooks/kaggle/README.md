# Kaggle Notebooks

This folder contains notebooks intended to run in Kaggle runtimes.

## Files

- `kaggle_phase1_setup_train.ipynb`: clone/install/setup plus initial training.
- `kaggle_phase1_eval_infer.ipynb`: baseline and adapted-model evaluation/inference.
- `kaggle_phase1_complete.ipynb`: complete Phase 1 Kaggle workflow.

## Expected Dataset

```text
klein2111/scannet-2d
```

The notebooks include dataset discovery and normalization because Kaggle-hosted dataset layouts can differ from local ScanNet-style assumptions.

## Output Policy

Do not commit Kaggle runtime outputs directly. Summarize final results in markdown reports under `results/reports/` or beside a specific experiment notebook.
