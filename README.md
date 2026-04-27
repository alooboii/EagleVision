# EagleVision

EagleVision is a modular research repository for improving a monocular depth estimator by training it inside a geometry-first round-trip novel-view-synthesis loop on indoor scenes. In Phase 1, the main model being improved is the depth estimator, not a learned renderer. The geometry stack provides explicit supervision and evaluation for geometric usefulness.

The repository preserves the existing Depth Anything V2 baseline under `baseline/depth_anything_v2` as a vendor-style dependency. New work lives under `src/eaglevision`, with configs, scripts, tests, and docs organized around a clean Phase 1 training and evaluation workflow.

## Phase 1 Focus

- load ScanNet-style paired indoor views with controlled camera motion
- render nearby target views with explicit RGB-D geometry
- adapt a mostly frozen Depth Anything V2 model with a lightweight residual head
- fuse warped and predicted target depth with simple visibility-based hole filling
- reconstruct the source view by backward warping
- train with round-trip rendering and depth anchoring losses
- evaluate both geometric usefulness and direct depth quality

## Repository Layout

```text
baseline/depth_anything_v2     Vendor baseline kept intact
configs/                       Data, model, train, and eval YAMLs
docs/                          Architecture, scope, training, and experiment notes
scripts/                       Thin command wrappers
src/eaglevision/               Phase 1 implementation
tests/                         Fast geometry and interface tests
```

## Install

```bash
pip install -e .[dev]
```

## Baseline CLI

The original baseline remains available:

```bash
python -m baseline.depth_anything_v2 download
python -m baseline.depth_anything_v2 infer --input <image-or-folder> --output-dir outputs/baseline
```

## Phase 1 CLI

Train:

```bash
python -m eaglevision.cli.train --config configs/train/phase1.yaml
```

Evaluate:

```bash
python -m eaglevision.cli.eval --config configs/eval/default.yaml
python -m eaglevision.cli.eval --config configs/eval/default.yaml --baseline-only
```

Fast-turn (roughly hour-scale) profile:

```bash
python -m eaglevision.cli.train --config configs/train/phase1_hourly.yaml
python -m eaglevision.cli.eval --config configs/eval/hourly.yaml --baseline-only
python -m eaglevision.cli.eval --config configs/eval/hourly.yaml --checkpoint <ckpt.pt>
```

The hourly profile bounds runtime with:

- lower resolution (`160x224`)
- smaller model (`vits`)
- frame subsampling (`frame_stride`)
- per-scene frame/pair caps (`max_frames_per_scene`, `max_pairs_per_scene`)
- capped training/eval work (`max_steps_per_epoch`, `eval.max_batches`)

Single-image depth inference:

```bash
python -m eaglevision.cli.infer_depth --input image.jpg --output outputs/depth.png
```

Geometric render from RGB-D:

```bash
python -m eaglevision.cli.render_novel_view \
  --rgb image.jpg \
  --depth depth.npy \
  --intrinsics intrinsics.txt \
  --transform transform.txt \
  --output outputs/novel_view.png
```

## Data Assumptions

Phase 1 expects ScanNet-style scene folders with `color/`, `depth/`, and `pose/` subdirectories. Pairing thresholds are config-driven and default to small reliable viewpoint changes:

- max translation: `0.30 m`
- max rotation: `10 deg`
- avoid near-identical views with minimum motion thresholds

See [docs/repository_guide.md](/C:/Users/Omore/OneDrive/Desktop/EagleVision/docs/repository_guide.md) and [src/eaglevision/data/README.md](/C:/Users/Omore/OneDrive/Desktop/EagleVision/src/eaglevision/data/README.md).

## Documentation

- [Architecture](/C:/Users/Omore/OneDrive/Desktop/EagleVision/docs/architecture.md)
- [Training Plan](/C:/Users/Omore/OneDrive/Desktop/EagleVision/docs/training_plan.md)
- [Experiments](/C:/Users/Omore/OneDrive/Desktop/EagleVision/docs/experiments.md)
- [Repository Guide](/C:/Users/Omore/OneDrive/Desktop/EagleVision/docs/repository_guide.md)
- [Phase 1 Scope](/C:/Users/Omore/OneDrive/Desktop/EagleVision/docs/phase1_scope.md)
- [Kaggle Submission Guide](/C:/Users/Omore/OneDrive/Desktop/EagleVision/docs/kaggle_submission.md)

## Kaggle

Submission-ready notebooks are available in [notebooks](/C:/Users/Omore/OneDrive/Desktop/EagleVision/notebooks).

Recommended Kaggle input dataset:

- `klein2111/scannet-2d`
- `https://www.kaggle.com/datasets/klein2111/scannet-2d`

Notebook entrypoints:

- [kaggle_phase1_setup_train.ipynb](/C:/Users/Omore/OneDrive/Desktop/EagleVision/notebooks/kaggle_phase1_setup_train.ipynb)
- [kaggle_phase1_eval_infer.ipynb](/C:/Users/Omore/OneDrive/Desktop/EagleVision/notebooks/kaggle_phase1_eval_infer.ipynb)
