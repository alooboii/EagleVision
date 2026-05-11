# EagleVision

EagleVision is a research codebase for improving monocular depth estimation through lightweight geometric adaptation. The project starts from a pretrained Depth Anything V2-style monocular depth estimator and adapts it for indoor RGB-D geometry using paired-view training, explicit camera geometry, and round-trip novel-view-synthesis supervision.

The current repository is organized around Phase 1: keep the pretrained depth backbone mostly frozen, train a lightweight residual adapter, and evaluate whether adaptation improves both direct depth quality and geometric usefulness compared with the frozen baseline.

## Project Overview

Monocular depth models are strong generic predictors, but their outputs are not always optimal for downstream 3D geometry. EagleVision studies whether a pretrained monocular depth model can be improved for RGB-D indoor scenes by training it inside a geometry-first adaptation loop.

The Phase 1 pipeline:

1. Loads ScanNet-style indoor RGB-D scenes with color images, metric depth, and camera poses.
2. Samples nearby source/target view pairs with controlled camera motion.
3. Predicts depth using a pretrained Depth Anything V2-style baseline plus a lightweight adapter.
4. Uses explicit camera intrinsics and poses to warp or render between nearby views.
5. Fuses warped depth and predicted depth with visibility-aware logic.
6. Trains with direct depth anchoring, reconstruction, and round-trip geometric consistency losses.
7. Evaluates against the frozen pretrained baseline using depth and geometric metrics.

The strongest current empirical result is summarized in the ablation report: the adapted model improves over the frozen baseline under a reduced-budget validation setup, with the best overall configuration identified as `lr_5e-5`.

See [notebooks/ablations/eaglevision_ablation_report.md](notebooks/ablations/eaglevision_ablation_report.md) for the final ablation interpretation.

## Research Focus

Phase 1 focuses on:

- ScanNet-style paired indoor views
- controlled translation and rotation thresholds for pair sampling
- explicit RGB-D geometric rendering instead of a learned renderer
- lightweight residual depth adaptation
- visibility-aware depth fusion
- direct metric depth supervision
- round-trip reconstruction and geometric consistency
- baseline-vs-adapted evaluation

Later phases can add stronger fusion, refinement, learned completion, source-depth prediction regimes, or generative components without disturbing the Phase 1 geometry stack.

## Directory Structure

```text
EagleVision/
  .github/
    ISSUE_TEMPLATE/             GitHub issue templates
    workflows/                  GitHub Actions CI
    pull_request_template.md    PR checklist

  baseline/
    depth_anything_v2/          Preserved Depth Anything V2-style baseline wrapper

  configs/
    data/                       Dataset configs
    eval/                       Evaluation configs
    model/                      Model and adapter configs
    train/                      Training configs

  docs/
    architecture.md             System architecture notes
    artifact_policy.md          Data/checkpoint/output policy
    experiments.md              Experiment notes
    kaggle_submission.md        Kaggle execution guide
    phase1_scope.md             Phase 1 boundaries
    repository_guide.md         Repo usage guide
    reproducibility.md          Reproduction workflow
    training_plan.md            Training plan
    eaglevision_paper_template.tex
    eaglevision_refs.bib

  notebooks/
    ablations/                  Ablation notebooks and ablation report
    archive/                    Older or duplicate notebooks kept for traceability
    evaluation/                 Evaluation and comparison notebooks
    experiments/                Active exploratory training notebooks
    kaggle/                     Kaggle setup/train/eval notebooks
    phase1_adapted_longrun/     Long-run adapted-only experiment

  outputs/                      Local generated outputs; ignored except docs/placeholders

  results/
    notebooks/                  Result-inspection notebooks
    reports/                    Small markdown experiment reports

  scripts/                      Thin wrappers around package CLIs

  src/
    eaglevision/
      cli/                      Train/eval/infer/render/demo entrypoints
      data/                     Dataset, transforms, pair sampler, collation
      engine/                   Trainer, evaluator, checkpointing, logging
      losses/                   Depth, photometric, and total losses
      metrics/                  Depth, image, and consistency metrics
      models/                   Adapter, depth wrapper, fusion, NVS, refinement
      utils/                    Geometry, intrinsics, I/O, masks, seeding
      visualization/            Panels and overlays

  tests/                        Fast CPU-friendly unit tests

  CONTRIBUTING.md
  README.md
  pyproject.toml
```

## Main Work Notebooks

These are the main notebooks a reader should inspect first.

| Notebook | Purpose |
|---|---|
| [notebooks/experiments/eaglevision2.ipynb](notebooks/experiments/eaglevision2.ipynb) | Main active training and evaluation notebook for the current EagleVision pipeline. |
| [notebooks/ablations/kaggle-ablations.ipynb](notebooks/ablations/kaggle-ablations.ipynb) | Main Kaggle ablation notebook used for the final reduced-budget hyperparameter ablation study. |
| [notebooks/ablations/eaglevision_hyperparameter_ablation_local_cuda_v3_unicode_single_progress.ipynb](notebooks/ablations/eaglevision_hyperparameter_ablation_local_cuda_v3_unicode_single_progress.ipynb) | Local/CUDA variant of the ablation workflow. |
| [notebooks/ablations/eaglevision_ablation_report.md](notebooks/ablations/eaglevision_ablation_report.md) | Final written ablation report explaining the ablation notebook results. |
| [notebooks/kaggle/kaggle_phase1_setup_train.ipynb](notebooks/kaggle/kaggle_phase1_setup_train.ipynb) | Kaggle setup and training notebook. |
| [notebooks/kaggle/kaggle_phase1_eval_infer.ipynb](notebooks/kaggle/kaggle_phase1_eval_infer.ipynb) | Kaggle evaluation and inference notebook. |
| [notebooks/kaggle/kaggle_phase1_complete.ipynb](notebooks/kaggle/kaggle_phase1_complete.ipynb) | Complete Phase 1 Kaggle workflow notebook. |
| [notebooks/evaluation/eval-complete.ipynb](notebooks/evaluation/eval-complete.ipynb) | Complete evaluation workflow notebook. |
| [notebooks/evaluation/eval-suite.ipynb](notebooks/evaluation/eval-suite.ipynb) | Evaluation suite notebook. |
| [notebooks/evaluation/eval_complete/eval-complete-final.ipynb](notebooks/evaluation/eval_complete/eval-complete-final.ipynb) | Final evaluation notebook from the complete evaluation folder. |
| [notebooks/evaluation/multi_model_multi_dataset_depth_benchmark.ipynb](notebooks/evaluation/multi_model_multi_dataset_depth_benchmark.ipynb) | Multi-model, multi-dataset depth benchmark notebook. |
| [notebooks/evaluation/quick-check-multi-dataset.ipynb](notebooks/evaluation/quick-check-multi-dataset.ipynb) | Quick multi-dataset sanity check notebook. |
| [notebooks/evaluation/quick_adapted_depth_preview.ipynb](notebooks/evaluation/quick_adapted_depth_preview.ipynb) | Quick qualitative preview of adapted depth predictions. |
| [notebooks/phase1_adapted_longrun/ablated-full-run.ipynb](notebooks/phase1_adapted_longrun/ablated-full-run.ipynb) | Long-run adapted-only Phase 1 experiment notebook. |

Archived root-level notebooks are preserved under [notebooks/archive/root-notebooks](notebooks/archive/root-notebooks) for traceability, but they are not the recommended entrypoints for new work.

## Key Reports

- [notebooks/ablations/eaglevision_ablation_report.md](notebooks/ablations/eaglevision_ablation_report.md): final ablation report and project justification.
- [results/reports/100ep_vs_Base.md](results/reports/100ep_vs_Base.md): 100-epoch adapted run compared with the baseline.
- [results/reports/post_ablation_training_100ep.md](results/reports/post_ablation_training_100ep.md): post-ablation 100-epoch training summary.

## Installation

Use Python 3.11 or 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the fast test suite:

```bash
pytest
```

## Main Commands

Train:

```bash
python -m eaglevision.cli.train --config configs/train/phase1.yaml
```

Evaluate:

```bash
python -m eaglevision.cli.eval --config configs/eval/default.yaml
python -m eaglevision.cli.eval --config configs/eval/default.yaml --baseline-only
```

Fast-turn profile:

```bash
python -m eaglevision.cli.train --config configs/train/phase1_hourly.yaml
python -m eaglevision.cli.eval --config configs/eval/hourly.yaml --baseline-only
python -m eaglevision.cli.eval --config configs/eval/hourly.yaml --checkpoint <ckpt.pt>
```

Single-image depth inference:

```bash
python -m eaglevision.cli.infer_depth --input image.jpg --output outputs/depth.png
```

Novel-view render from RGB-D:

```bash
python -m eaglevision.cli.render_novel_view \
  --rgb image.jpg \
  --depth depth.npy \
  --intrinsics intrinsics.txt \
  --transform transform.txt \
  --output outputs/novel_view.png
```

Console scripts after installation:

```bash
eaglevision-train
eaglevision-eval
eaglevision-compare
eaglevision-infer-depth
eaglevision-render
eaglevision-demo
```

## Data

Phase 1 expects ScanNet-style scene folders:

```text
scene_id/
  color/
  depth/
  pose/
```

Default pair sampling is conservative:

- max translation: `0.30 m`
- max rotation: `10 deg`
- minimum motion thresholds to avoid near-identical pairs

Data should stay outside git. Use local `data/`, Kaggle inputs, or mounted storage, then point configs at the dataset root. See [src/eaglevision/data/README.md](src/eaglevision/data/README.md) and [docs/kaggle_submission.md](docs/kaggle_submission.md).

## Baseline

The baseline integration is isolated under [baseline/depth_anything_v2](baseline/depth_anything_v2). It remains available through:

```bash
python -m baseline.depth_anything_v2 download
python -m baseline.depth_anything_v2 infer --input <image-or-folder> --output-dir outputs/baseline
```

## Repository Rules

- Commit source, configs, docs, tests, and small markdown reports.
- Do not commit raw datasets, generated outputs, local run folders, or large checkpoint artifacts.
- Use Git LFS only for deliberately retained model artifacts.
- Keep notebooks in the folder matching their role.
- Keep ablation reports beside the ablation notebooks.
- Keep `results/` for small reports and result-inspection notebooks, not raw experiment dumps.

See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/repository_guide.md](docs/repository_guide.md) for contributor workflow details.

## Documentation Index

- [Repository Guide](docs/repository_guide.md)
- [Architecture](docs/architecture.md)
- [Training Plan](docs/training_plan.md)
- [Experiments](docs/experiments.md)
- [Phase 1 Scope](docs/phase1_scope.md)
- [Kaggle Submission Guide](docs/kaggle_submission.md)
- [Artifact Policy](docs/artifact_policy.md)
- [Reproducibility](docs/reproducibility.md)
