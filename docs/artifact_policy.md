# Artifact Policy

EagleVision is an ML research repository, so artifact boundaries must be explicit. The repository should stay useful on GitHub without becoming a dump of raw datasets, generated outputs, or large training products.

## Commit These

- source code under `src/`
- tests under `tests/`
- reusable configs under `configs/`
- project documentation
- small markdown reports
- curated notebooks that are useful for reproduction or review
- small placeholders such as `.gitkeep`

## Do Not Commit These

- raw datasets
- generated `outputs/`
- local run folders
- cache directories
- credentials or private tokens
- large checkpoints unless intentionally tracked with Git LFS
- uncurated image/video/array dumps

## Checkpoints

Use one of these options for checkpoints:

- local ignored `checkpoints/` for development
- GitHub Releases for public release artifacts
- Hugging Face Hub for model distribution
- Kaggle datasets for Kaggle notebook dependencies
- Git LFS only for deliberately retained files

Current special case:

- `notebooks/phase1_adapted_longrun/best_100ep.pt` is retained as the selected long-run checkpoint artifact/pointer under the repository's Git LFS policy.

## Notebooks

Notebook folders should communicate intent:

- `notebooks/ablations/`: ablation notebooks and ablation report
- `notebooks/kaggle/`: Kaggle workflows
- `notebooks/evaluation/`: evaluation workflows and comparisons
- `notebooks/experiments/`: active exploratory work
- `notebooks/archive/`: older or duplicate notebooks kept only for traceability
- `results/notebooks/`: result-review notebooks and rendered report README files

Strip bulky transient notebook output unless the output is central to the report being reviewed.

## Reports

Small markdown reports belong in one of three places:

- `notebooks/ablations/` when the report directly explains an ablation notebook
- `results/notebooks/README.md` when GitHub should render a folder-level evaluation report
- `results/reports/` for standalone written summaries

Generated images, raw metric dumps, and bulky outputs should stay in ignored output folders unless a small curated artifact is needed for a paper or report.
