# Contributing

This repository is a research codebase, so contributions should preserve both code quality and experiment traceability.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Before Opening A Pull Request

Run:

```bash
pytest
```

If you changed formatting-sensitive Python code, also run:

```bash
ruff check .
```

For notebook changes, strip bulky execution output unless the output is the actual artifact being reviewed:

```bash
nbstripout notebooks/path/to/notebook.ipynb
```

## What Belongs In Git

Commit:

- source code under `src/`
- tests under `tests/`
- reusable configs under `configs/`
- markdown documentation and small reports
- curated notebooks that are useful for reproduction

Do not commit:

- raw datasets
- generated `outputs/`
- local run directories
- large checkpoints unless intentionally tracked with Git LFS
- credentials, Kaggle tokens, API keys, or machine-specific paths

## Notebook Policy

Put notebooks in the folder matching their purpose:

- `notebooks/ablations/` for ablation runs and the ablation report
- `notebooks/kaggle/` for Kaggle-facing workflows
- `notebooks/evaluation/` for evaluation and comparison work
- `notebooks/experiments/` for active research notebooks
- `notebooks/archive/` for older notebooks kept only for traceability

## Pull Request Checklist

- Summary explains the change and why it is needed.
- Tests or validation commands are listed.
- Any data, checkpoint, or artifact impact is documented.
- Notebook moves preserve links from README files.
- New configs explain their intended runtime profile.
