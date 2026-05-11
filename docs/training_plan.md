# Training Plan

The primary Phase 1 training mode is `roundtrip_gt_source_depth`.

This mode uses ground-truth source depth for the forward warp, then trains the adapted depth estimator through target-view and round-trip constraints.

## Training Graph

```text
source RGB + source GT depth
        |
        v
forward geometric warp into target camera
        |
        v
warped target RGB + warped target depth + valid/hole masks
        |
        v
DepthAnythingWithAdapter predicts target depth
        |
        v
fuse warped depth with predicted depth
        |
        v
backward geometric warp into source camera
        |
        v
reconstructed source RGB/depth
        |
        v
losses: target RGB, cycle RGB, cycle depth, target depth anchoring
```

## Default Loss Weights

| Loss | Weight | Role |
|---|---:|---|
| `target_rgb` | 1.0 | Keeps the forward target warp visually aligned with the target frame. |
| `cycle_rgb` | 1.0 | Keeps the backward source reconstruction visually aligned with the source frame. |
| `cycle_depth` | 0.5 | Encourages source-depth consistency after the round trip. |
| `target_depth` | 0.25 | Anchors target prediction to metric depth when target GT depth is available. |

## Runtime Profiles

| Config | Purpose |
|---|---|
| `configs/train/phase1.yaml` | Default local profile, CPU by default, `vitl`, 240x320. |
| `configs/train/phase1_hourly.yaml` | Faster CUDA profile, `vits`, 160x224, capped steps and eval batches. |
| `configs/train/phase1_kaggle_complete.yaml` | Kaggle-oriented profile, CUDA, `vits`, 240x320. |

## Training Outputs

Training writes under the configured `output_dir`, usually inside `outputs/`.

Expected outputs:

- `train_metrics.jsonl`
- debug panels under `panels/`
- periodic checkpoints under `checkpoints/`
- final checkpoint for each epoch

These generated outputs are ignored by git by default. Promote only small curated summaries into `results/` or markdown documentation.

## Practical Workflow

1. Start with `phase1_hourly.yaml` to validate dataset paths, pair sampling, CUDA availability, logs, panels, and checkpoints.
2. Compare baseline-only evaluation against adapted evaluation on the same config/split.
3. Use Kaggle notebooks for larger runs when local hardware is insufficient.
4. Summarize final results in markdown, not only notebook output.

## Current Training Interpretation

The 100-epoch adapted run improves several depth/geometric metrics over the frozen baseline. It does not fully solve full-frame image reconstruction because hole coverage and invalid projected regions still hurt full-image PSNR/SSIM.

Future training work should focus on:

- improving projection coverage and hole handling
- testing learned fusion or refinement
- tightening pose/depth-scale sanity checks
- preserving gains in depth metrics while improving valid/full image metrics
