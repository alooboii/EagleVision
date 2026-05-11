# Documentation

This folder contains the project-level documentation for EagleVision. Folder-level `README.md` files explain local directories; this `docs/` folder explains the research system, workflow, scope, and reproducibility rules.

## Main Docs

| File | Purpose |
|---|---|
| `architecture.md` | Detailed Phase 1 architecture overview: data flow, `RoundTripDepthNVS`, model modules, training/evaluation architecture, config layout, and extension points. |
| `phase1_scope.md` | Current claim boundary: what Phase 1 includes, excludes, and can safely claim. |
| `training_plan.md` | Training graph, default loss weights, runtime profiles, outputs, and next training priorities. |
| `experiments.md` | Current experiment families, reports, notebook artifacts, and recommended comparisons. |
| `repository_guide.md` | Practical guide to the top-level repo layout, source layout, notebook layout, reports, commands, and artifact rules. |
| `kaggle_submission.md` | Kaggle dataset/runtime setup notes. |
| `artifact_policy.md` | Rules for data, checkpoints, outputs, reports, and notebooks. |
| `reproducibility.md` | Clean-clone verification, training, evaluation, ablation, Kaggle, and report reproduction steps. |

## Paper Assets

| File | Purpose |
|---|---|
| `eaglevision_paper_template.tex` | LaTeX paper template. |
| `eaglevision_refs.bib` | Bibliography file. |

## Key Report Locations

- Complete evaluation report: `results/notebooks/README.md`
- Copied complete evaluation notebook: `results/notebooks/eval-complete-final.ipynb`
- Final ablation report: `notebooks/ablations/eaglevision_ablation_report.md`

Keep the root `README.md` linked to any documentation that a first-time reader needs.
