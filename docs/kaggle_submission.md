# Kaggle Submission Guide

## Recommended Kaggle Dataset

Best Kaggle-hosted ScanNet-related option found for this repository:

- `klein2111/scannet-2d`
- `https://www.kaggle.com/datasets/klein2111/scannet-2d`

Why this one:

- the project targets indoor RGB-D geometry with paired views and camera poses
- among Kaggle-discoverable options, this is the closest ScanNet-branded match
- it is more aligned with the project than generic RGB-D scene or synthetic depth datasets

Practical caveat:

- the public Kaggle page provides limited structure and license detail
- the notebooks therefore perform dataset inspection and normalization before training
- if its layout differs from the expected ScanNet-style scene structure, adjust the dataset discovery helper in the notebook once

## Kaggle Workflow

1. Create a Kaggle notebook with internet enabled.
2. Add `klein2111/scannet-2d` to notebook inputs.
3. Run `notebooks/kaggle_phase1_setup_train.ipynb` for setup and training.
4. Export training outputs as a Kaggle dataset if you want a separate evaluation notebook.
5. Run `notebooks/kaggle_phase1_eval_infer.ipynb` for evaluation and inference.

## Default Kaggle Input Paths

The notebooks default to:

```python
RAW_DATASET_DIR = Path('/kaggle/input/scannet-2d')
TRAIN_RUN_INPUT_DIR = Path('/kaggle/input/eaglevision-phase1-run')
```

## Expected Normalized Scene Layout

The repository expects each scene to normalize into:

```text
sceneXXXX_YY/
  color/
  depth/
  pose/
```

The notebooks support:

- attached folders
- `.zip` archives
- `.tar.gz` / `.tgz` archives

## Submission Checklist

- root README explains project framing and commands
- docs cover architecture, scope, experiments, repository layout, and Kaggle usage
- Kaggle notebooks show a real end-to-end workflow
- tracked configs remain generic, while Kaggle-specific configs are generated into `outputs/kaggle_configs/`
