# Artifact Policy

EagleVision is an ML research repository, so artifact boundaries must be explicit.

## Commit These

- source code
- tests
- reusable configs
- project documentation
- small markdown reports
- curated notebooks that help reproduce project results

## Do Not Commit These

- raw datasets
- generated `outputs/`
- local run folders
- cache directories
- credentials or private tokens
- large checkpoints unless intentionally tracked with Git LFS

## Checkpoints

Use one of these options for checkpoints:

- local ignored `checkpoints/` for development
- GitHub Releases for public release artifacts
- Hugging Face Hub for model distribution
- Kaggle datasets for Kaggle notebook dependencies
- Git LFS only for deliberately retained files

## Results

Small markdown reports belong in `results/reports/`.

Ablation-specific reports belong beside the ablation notebooks in `notebooks/ablations/` so readers can inspect the notebook and written interpretation together.

Generated images, raw metric dumps, and bulky outputs should stay in ignored output folders unless a small curated artifact is needed for a paper or report.
