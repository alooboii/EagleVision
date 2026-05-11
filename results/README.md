# Results

This folder contains small result artifacts that are useful to review in GitHub.

## Folder Layout

| Folder | Purpose |
|---|---|
| `notebooks/` | Result-inspection notebooks and the complete evaluation report README. |
| `reports/` | Small standalone markdown summaries. |

## Key Files

| File | Purpose |
|---|---|
| `notebooks/README.md` | Complete evaluation report rendered on GitHub. |
| `notebooks/eval-complete-final.ipynb` | Copied final evaluation notebook used by the report. |
| `notebooks/check.ipynb` | Older result checking notebook. |
| `notebooks/eval.ipynb` | Older evaluation result notebook. |
| `reports/100ep_vs_Base.md` | 100-epoch adapted run compared with the baseline. |
| `reports/post_ablation_training_100ep.md` | Post-ablation 100-epoch training summary. |

## Policy

Keep bulky generated assets out of this folder unless they are intentionally curated. Raw outputs, checkpoints, images, and metric dumps should stay under ignored runtime folders such as `outputs/`.

The final ablation report lives in `notebooks/ablations/` because it directly explains the ablation notebooks. The complete evaluation report lives here because GitHub renders `results/notebooks/README.md` when viewing the result notebook folder.
