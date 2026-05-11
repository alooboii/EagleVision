# Results

This folder contains small result artifacts that are useful to review in GitHub.

## Folder Layout

- `reports/`: markdown summaries, experiment reports, and final written comparisons.
- `notebooks/`: result notebooks kept for traceability, including the complete evaluation report README.

## Policy

Keep bulky generated assets out of this folder unless they are intentionally curated. Raw outputs, checkpoints, images, and metric dumps should stay under ignored runtime folders such as `outputs/`.

The final ablation report lives in `notebooks/ablations/` because it directly explains the ablation notebooks. The complete evaluation report lives in `results/notebooks/README.md` beside the copied `eval-complete-final.ipynb` notebook.
