# Architecture

EagleVision Phase 1 is a geometry-first depth adaptation system. The main object being improved is the monocular depth estimator, not a learned renderer. Rendering is kept explicit and interpretable so depth changes can be evaluated through geometric behavior.

## System Overview

```text
ScanNet-style RGB-D pair
  source RGB, source depth, source pose, target RGB, target depth, target pose, intrinsics
        |
        v
ScanNetPairDataset + PairSamplingConfig
        |
        v
RoundTripDepthNVS
  1. source RGB + source GT depth -> target-view forward warp
  2. target-view warped RGB -> DepthAnythingWithAdapter
  3. warped target depth + predicted target depth -> visibility-aware fusion
  4. fused target RGB-D -> source-view backward warp
  5. reconstructed source RGB -> adapted source-depth prediction
        |
        v
Phase 1 losses and metrics
  target RGB, cycle RGB, cycle depth, target depth anchoring, reprojection/depth metrics
```

## Main Runtime Modules

| Area | Path | Responsibility |
|---|---|---|
| Dataset and pairs | `src/eaglevision/data/` | ScanNet-style scene discovery, paired-view samples, transforms, collation, and motion-constrained pair filtering. |
| Depth model | `src/eaglevision/models/depth/` | Wraps Depth Anything V2-style checkpoints and exposes base/adapted depth predictions. |
| Adapter | `src/eaglevision/models/adapters.py` | Lightweight residual depth adapter trained while the backbone is mostly frozen. |
| Round-trip system | `src/eaglevision/models/rt_depthnvs.py` | Composes depth prediction, warping, fusion, and source reconstruction. |
| NVS geometry | `src/eaglevision/models/nvs/` | Backprojection, projection, z-buffer style rasterization, geometric warping, and visibility masks. |
| Depth fusion | `src/eaglevision/models/fusion/` | Fuses warped depth with predicted depth using visibility/hole-aware logic. |
| Losses | `src/eaglevision/losses/` | Target RGB, cycle RGB, cycle depth, target depth, and total objective composition. |
| Metrics | `src/eaglevision/metrics/` | Depth, image, and geometric consistency measurements. |
| Engine | `src/eaglevision/engine/` | Training loop, evaluation loop, checkpointing, and JSONL metric logging. |
| CLIs | `src/eaglevision/cli/` | Train, evaluate, compare, infer depth, render novel views, and demo round trips. |

## Data Flow

The core model class is `RoundTripDepthNVS`.

Input batch keys:

- `source_rgb`
- `target_rgb`
- `source_depth`
- `target_depth`
- `source_intrinsics`
- `target_intrinsics`
- `source_pose`
- `target_pose`

The model computes:

1. `T_s2t = target_pose @ inverse(source_pose)`
2. `T_t2s = source_pose @ inverse(target_pose)`
3. forward target warp from source RGB-D
4. adapted target depth from the warped target RGB
5. fused target depth from valid warped depth and predicted depth
6. backward source reconstruction from fused target RGB-D
7. adapted source-depth prediction from the reconstructed source RGB

Important output keys:

- `A`: source RGB
- `B`: target RGB
- `D_source_gt`: source ground-truth depth
- `D_target_gt`: target ground-truth depth when available
- `A_t_warp`: source RGB forward-warped into target view
- `D_t_warp`: source depth forward-warped into target view
- `M_t`: valid target projection mask
- `H_t`: target hole mask
- `D_t_pred`: adapted predicted target depth
- `D_t_pred_base`: frozen baseline target depth
- `D_t_fused`: fused target depth
- `A_s_recon`: source RGB reconstructed by backward warp
- `D_s_warp`: fused target depth warped back to source
- `D_s_pred`: adapted depth on reconstructed source
- `M_s`: valid source reconstruction mask
- `H_s`: source hole mask

## Training Architecture

The training CLI is `python -m eaglevision.cli.train --config <config>`.

The trainer:

1. loads YAML config
2. sets seed and device
3. builds train/validation `ScanNetPairDataset` instances
4. creates `DepthAnythingWithAdapter`
5. wraps it in `RoundTripDepthNVS`
6. optimizes trainable parameters with AdamW
7. writes metrics, debug panels, and checkpoints under `outputs/`

Default Phase 1 loss weights:

| Loss | Weight | Purpose |
|---|---:|---|
| `target_rgb` | 1.0 | Encourage source-to-target warp consistency. |
| `cycle_rgb` | 1.0 | Encourage target-to-source round-trip image consistency. |
| `cycle_depth` | 0.5 | Keep reconstructed source depth consistent with source GT depth. |
| `target_depth` | 0.25 | Anchor predicted target depth to metric depth when available. |

## Evaluation Architecture

The evaluation CLI is `python -m eaglevision.cli.eval --config <config>`.

Evaluation computes:

- target RGB L1
- cycle RGB L1
- PSNR
- SSIM
- reprojection depth consistency
- depth L1
- RMSE
- AbsRel
- total configured losses

The complete final evaluation notebook adds a larger dashboard around this core behavior, including paired baseline-vs-adapted tests, motion buckets, scale diagnostics, runtime/memory summaries, qualitative galleries, and plots. The rendered report is in `results/notebooks/README.md`, and the copied source notebook is `results/notebooks/eval-complete-final.ipynb`.

## Config Architecture

Reusable runtime profiles live under `configs/`.

- `configs/train/phase1.yaml`: default CPU/local Phase 1 profile.
- `configs/train/phase1_hourly.yaml`: reduced-cost CUDA profile with smaller model/resolution and capped steps.
- `configs/train/phase1_kaggle_complete.yaml`: Kaggle-oriented Phase 1 profile.
- `configs/eval/default.yaml`: default local evaluation.
- `configs/eval/hourly.yaml`: reduced-cost CUDA evaluation.
- `configs/eval/phase1_kaggle_complete.yaml`: Kaggle-oriented evaluation.
- `configs/model/depth_adapter.yaml`: standalone depth adapter settings.
- `configs/model/roundtrip.yaml`: round-trip model mode and fusion type.
- `configs/data/scannet.yaml`: ScanNet-style data defaults.

## Baseline Boundary

`baseline/depth_anything_v2/` is intentionally isolated. It preserves baseline model loading, downloading, preprocessing, and inference behavior. New EagleVision research logic should live under `src/eaglevision/`.

This separation makes the comparison clean:

- baseline package: frozen pretrained model behavior
- EagleVision package: adapter, geometry loop, training/evaluation, metrics, and reports

## Extension Points

The architecture is designed so later phases can add:

- learned depth fusion
- source-depth prediction regimes
- refinement networks
- generative disocclusion completion
- stronger pair sampling and scene split controls
- richer evaluation dashboards

Those extensions should preserve the explicit geometry backbone unless the research question changes.
