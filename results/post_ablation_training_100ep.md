# Phase 1 Adapted Long-Run Experiment

This folder contains a partner-added Kaggle long-run experiment for EagleVision Phase 1.

## What This Notebook Does

`ablated-full-run.ipynb` runs an adapted-only Phase 1 EagleVision workflow on Kaggle. It:

- Clones and installs the EagleVision repo inside Kaggle.
- Inspects and normalizes a ScanNet-style dataset into the repo's expected `data/scannet` layout.
- Downloads Depth Anything V2 checkpoints.
- Builds Kaggle-local train/eval YAML configs.
- Trains a lightweight residual/depth adapter on top of Depth Anything V2.
- Freezes the Depth Anything V2 backbone and trains adapter parameters.
- Uses ScanNet-style RGB/depth/pose pairs for geometry-aware training/evaluation.
- Performs a checkpoint sweep across the 100-epoch run.
- Selects the best checkpoint using `abs_rel` as the primary metric.
- Saves adapted metrics, checkpoint sweep artifacts, plots, and inference/visualization outputs.

## Important Implementation Details

| Setting | Value |
|---|---:|
| Experiment tag | `phase1_kaggle_best_longrun_v1` |
| Depth mode | `metric` |
| Depth encoder | `vits` |
| Metric profile | `hypersim` |
| Backbone frozen | `True` |
| Adapter hidden channels | `32` |
| Dataset | ScanNet-style 2D dataset on Kaggle |
| Discovered scenes | `1513` |
| Train/validation split | `90% / 10%` |
| Image size | `[192, 288]` |
| Seed | `7` |
| Training epochs | `100` |
| Learning rate | `5e-5` |
| Weight decay | `1e-4` |
| Batch size | `1` |

Loss weights:

| Loss | Weight |
|---|---:|
| `target_rgb` | `1.0` |
| `cycle_rgb` | `1.0` |
| `cycle_depth` | `0.35` |
| `target_depth` | `0.35` |

Pairing settings:

| Setting | Value |
|---|---:|
| Min translation | `0.02 m` |
| Max translation | `0.30 m` |
| Min rotation | `0.8 degrees` |
| Max rotation | `8.0 degrees` |
| Max index gap | `10` |
| Frame stride | `2` |
| Max frames per scene | `180` |
| Max pairs per scene | `120` |

## Checkpoint Sweep

The run performs a coarse sweep every 5 epochs using 300 eval batches, then full evaluation on the top 5 candidates using 780 eval batches. The primary selection metric is `abs_rel`, where lower is better.

The selected checkpoint was epoch 85:

`epoch_085_final_step_0293590.pt`

The uploaded checkpoint file `best_100ep.pt` appears to correspond to this selected checkpoint because its saved step is `293590`.

## Final Selected Adapted Metrics

| Metric | Value | Direction |
|---|---:|---|
| target_rgb_l1 | 0.210844 | lower is better |
| cycle_rgb_l1 | 0.118281 | lower is better |
| psnr | 14.109756 | higher is better |
| ssim | 0.120327 | higher is better |
| reprojection_depth | 1.495880 | lower is better |
| depth_l1 | 2.592129 | lower is better |
| rmse | 2.754789 | lower is better |
| abs_rel | 0.317252 | lower is better |
| loss_total | 1.759928 | lower is better |

## Interpretation

This is useful progress for the project because it gives us a full 100-epoch adapted-only training run, a selected reusable checkpoint, and a repeatable Kaggle workflow. It is good as an engineering/reproducibility contribution.

It is not yet enough by itself to claim a final research improvement, because the notebook explicitly skips baseline evaluation by design. The results show the best adapted checkpoint within this run, but they do not prove improvement over the unadapted Depth Anything V2 baseline unless compared against a baseline using the same validation split and metrics.

The selected metric `abs_rel = 0.317252` is the best among the full-evaluation candidates. PSNR around `14.11` and SSIM around `0.12` are modest/low for image reconstruction quality, so the visual novel-view synthesis side still needs improvement.

The visual output shows that the forward warp / novel view has sparse holes and artifacts, which means this is not final-quality NVS yet.

Overall conclusion: good for our project as a long-run adapter checkpoint and experiment artifact, but not enough alone as proof of a strong final result. It should be integrated as a Phase 1 long-run experiment and later compared against baseline and other ablations.

## How to Use This Folder

- Open `ablated-full-run.ipynb` to inspect the Kaggle workflow.
- Use `best_100ep.pt` as the selected adapted checkpoint from the 100-epoch run.
- For future paper/reporting, compare this checkpoint against the baseline Depth Anything V2 model using the same validation split and same eval config.

## Next Steps

- Add baseline evaluation for the same validation scenes.
- Add side-by-side baseline vs adapted metrics.
- Add qualitative visual comparison: input RGB, baseline depth, adapted depth, novel view output.
- Add more NVS-specific quality metrics if available.
- Consider masking invalid warp pixels before calculating RGB metrics.
- Document exact Kaggle dataset version and checkpoint download links if not already included in the notebook.
