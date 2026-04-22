# Kaggle Notebooks

These notebooks are intended for Kaggle execution with the repository cloned inside the notebook runtime.

Recommended Kaggle input dataset:

- `klein2111/scannet-2d`
- `https://www.kaggle.com/datasets/klein2111/scannet-2d`

Why this dataset:

- it is the clearest Kaggle-hosted ScanNet-derived option found for this project
- it is materially closer to the repository objective than generic RGB-D alternatives on Kaggle
- its scale appears sufficient for a real indoor paired-view geometry workflow

Important caveat:

- the Kaggle listing exposes limited structural documentation and unclear license metadata
- the notebooks therefore inspect and normalize the dataset before training
- if the folder structure differs from ScanNet-style scene folders, adjust the discovery helper in the notebook

Notebook roles:

- `kaggle_phase1_setup_train.ipynb`: clone repo, install deps, set up dataset, download checkpoint, generate configs, train, run baseline eval
- `kaggle_phase1_eval_infer.ipynb`: clone repo, set up dataset, run baseline/adapted eval, compare metrics, run inference
