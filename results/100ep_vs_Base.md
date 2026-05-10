# Complete EagleVision Evaluation Suite

This folder contains the completed Kaggle evaluation notebook:

`eval-complete-final.ipynb`

The notebook is a full evaluation and diagnostic report for EagleVision Phase 1. It is not a training notebook. It evaluates whether the adapted EagleVision checkpoint improves over the raw Depth Anything V2 baseline under the same validation scenes, same source-target pairs, same masks, and same metrics.

## Why This Notebook Exists

EagleVision uses Depth Anything V2 with a lightweight adapter for indoor RGB-D geometry and depth-assisted novel view synthesis. Earlier results showed that the adapted model could look visually reasonable while some metrics were weak. This notebook was created to diagnose whether those weak metrics come from actual model failure or from evaluation artifacts such as:

- black holes from forward warping
- invalid projected pixels
- occlusion and disocclusion
- low valid overlap between source and target views
- hard camera motion
- depth scale mismatch
- full-image PSNR/SSIM unfairly punishing regions that should be masked

The core question is:

> Does the adapted `best_100ep.pt` checkpoint beat the unadapted Depth Anything V2 baseline when evaluated fairly on identical pairs and visibility masks?

## Compared Methods

The notebook compares four methods:

| Method | Purpose |
|---|---|
| Identity baseline | Copies source RGB as the target prediction. This measures how hard the source-target pair is. |
| GT-depth warp | Uses source RGB plus source ground-truth depth, camera poses, and intrinsics to warp into the target view. This is a geometry/data sanity upper bound. |
| DAV2 baseline | Uses raw unadapted Depth Anything V2 depth for geometric warping. |
| Adapted 100ep | Uses the partner 100-epoch adapted checkpoint and evaluates `adapted_depth`. |

## What The Notebook Does

The notebook:

- sets up the EagleVision repo on Kaggle
- downloads/checks the Depth Anything V2 `vits` Hypersim metric checkpoint
- discovers the ScanNet-style Kaggle dataset
- normalizes the dataset into the repo's expected `data/scannet` layout
- builds deterministic validation pairs using the Phase 1 long-run pairing settings
- loads the DAV2 baseline and adapted 100-epoch checkpoint
- evaluates identity, GT-depth warp, DAV2 baseline, and adapted checkpoint on the same pairs
- computes full-image, valid-overlap, hole-aware, edge-aware, and depth-scale metrics
- computes paired adapted-vs-baseline deltas
- creates per-scene diagnostics and failure rankings
- creates runtime and memory summaries
- saves plots, qualitative grids, CSVs, logs, and a generated report
- packages all outputs into `outputs/eval_suite_phase1.zip`

## Dataset And Checkpoints

The run is designed for the ScanNet-style Kaggle dataset:

`klein2111/scannet-2d`

The expected Kaggle scene root is:

`/kaggle/input/datasets/klein2111/scannet-2d/scannet_2d`

The adapted checkpoint is:

`best_100ep.pt`

The expected DAV2 checkpoint is:

`depth_anything_v2_metric_hypersim_vits.pth`

The notebook follows the partner ablation workflow for baseline setup:

```bash
python -m baseline.depth_anything_v2 download --mode all --profile hypersim --encoder vits
```

## Metrics

The notebook reports:

- RGB L1, PSNR, SSIM, optional LPIPS
- full-image metrics and valid-overlap metrics
- visible-region and hole-region metrics
- AbsRel, RMSE, delta accuracy, depth L1, SILog
- median-scaled AbsRel and scale-ratio diagnostics
- edge and non-edge depth AbsRel
- valid pixel ratio, hole ratio, overlap ratio
- reprojection depth L1
- motion bucket and difficulty score
- paired adapted-vs-DAV2 improvements
- bootstrap confidence intervals
- runtime and CUDA memory diagnostics

Full-image metrics measure final rendered image quality, including holes. Valid-overlap metrics measure only pixels reached by geometric projection and are fairer for comparing depth methods. Hole metrics diagnose how much full-image quality is being hurt by disoccluded or invalid regions. Depth-scale diagnostics show whether bad depth metrics are caused by global scale mismatch rather than local structure.

## Main Results From This Run

The completed notebook evaluated 433 source-target pairs, producing 1,732 method rows.

Main comparison table:

| Method | AbsRel ↓ | RMSE ↓ | Delta1 ↑ | PSNR Full ↑ | PSNR Valid ↑ | SSIM Full ↑ | SSIM Valid ↑ | Valid Ratio ↑ | Hole Ratio ↓ | Reproj Depth L1 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Adapted 100ep | 0.344948 | 2.187455 | 0.395220 | 8.030335 | 11.299679 | 0.079778 | 0.748439 | 0.403856 | 0.596144 | 2.020791 |
| DAV2 baseline | 0.461546 | 2.371236 | 0.312454 | 8.472656 | 11.418896 | 0.102726 | 0.704900 | 0.467533 | 0.532467 | 2.208435 |
| GT-depth warp | 2.300415 | 2.620927 | 0.563551 | 7.986739 | 11.555642 | 0.062490 | 0.782498 | 0.368082 | 0.631918 | 2.345473 |
| Identity baseline | NaN | NaN | NaN | 13.679151 | 13.679151 | 0.399479 | 0.399479 | 1.000000 | 0.000000 | NaN |

Paired adapted-vs-DAV2 summary:

| Metric | Mean Improvement | Median Improvement | Pairs Adapted Better |
|---|---:|---:|---:|
| abs_rel | 0.114954 | 0.048052 | 63.92% |
| rmse | 0.143124 | 0.073363 | 60.53% |
| delta1 | 0.082628 | 0.036248 | 60.05% |
| psnr_full | -0.442322 | -0.259539 | 29.56% |
| psnr_valid | -0.135256 | -0.055729 | 47.10% |
| ssim_full | -0.022948 | -0.012608 | 27.25% |
| ssim_valid | 0.039255 | 0.040862 | 74.64% |
| reprojection_depth_l1 | 0.149793 | 0.086295 | 62.95% |
| rgb_l1_valid | -0.005261 | -0.000631 | 47.83% |
| median_scaled_absrel | -0.012884 | -0.010033 | 39.23% |

Positive improvement means adapted is better for that metric, after accounting for metric direction.

## Interpretation

The adapted checkpoint improves several depth and geometry metrics relative to the raw DAV2 baseline:

- lower AbsRel
- lower RMSE
- higher delta1
- lower reprojection depth L1
- higher valid-overlap SSIM

However, it does not cleanly improve image reconstruction metrics:

- full-image PSNR is worse than DAV2
- valid-overlap PSNR is slightly worse than DAV2
- full-image SSIM is worse than DAV2
- valid RGB L1 is slightly worse
- adapted has lower valid pixel coverage and higher hole ratio than DAV2

This means the adapted model appears to improve depth/geometric consistency more than final RGB novel-view quality. The visual synthesis side is still limited by holes, sparse forward warps, and coverage.

The identity baseline has much higher PSNR/SSIM than the warped methods. This does not mean identity is a better NVS model. It means many selected source-target pairs are visually similar enough that copying the source image scores well under RGB reconstruction metrics. This is why the notebook includes motion buckets, difficulty scores, valid-overlap metrics, and GT-warp sanity checks.

The GT-depth warp result is a major caveat. It has strong delta1 and valid SSIM, but its average AbsRel is poor because of severe outliers. If GT-depth warp does not behave like a clean upper bound, then pose conventions, depth scale, intrinsics, sparse splatting, or invalid-depth handling should be checked before making final research claims from NVS metrics.

## Main Takeaway

The adapted 100-epoch checkpoint is useful and shows evidence of depth/geometry improvement over raw DAV2, especially in AbsRel, RMSE, delta1, and reprojection-depth metrics. It is not yet a strong final novel-view-synthesis result because full-image and PSNR-based image metrics remain weak, identity remains hard to beat on RGB metrics, and the GT-depth warp sanity result exposes geometry/evaluation issues that need follow-up.

For reporting, the safest claim is:

> The adapted checkpoint improves several depth and geometry metrics over the unadapted DAV2 baseline under paired evaluation, but image reconstruction quality and forward-warp completeness remain limited. The evaluation also indicates that geometry and visibility handling need further calibration before claiming strong final NVS performance.

## Outputs Produced

The notebook writes outputs under:

`outputs/eval_suite_phase1/`

Important files include:

- `metrics/metrics_per_pair.csv`
- `metrics/metrics_overall.csv`
- `metrics/metrics_by_method.csv`
- `metrics/metrics_by_motion_bucket.csv`
- `metrics/metrics_by_scene.csv`
- `metrics/metrics_by_difficulty.csv`
- `metrics/baseline_vs_adapted.csv`
- `metrics/adapted_vs_baseline_paired.csv`
- `metrics/depth_scale_diagnostics.csv`
- `metrics/runtime_memory_summary.csv`
- `diagnostics/sanity_ranking.json`
- `diagnostics/worst_scenes_by_absrel.csv`
- `diagnostics/worst_scenes_by_psnr_valid.csv`
- `diagnostics/worst_scenes_by_hole_ratio.csv`
- `diagnostics/best_scenes_overall.csv`
- `plots/`
- `qualitative/`
- `logs/eval_suite.log`
- `logs/pair_errors.jsonl`
- `report.md`
- `outputs/eval_suite_phase1.zip`

## Known Limitations

- Forward warping creates holes and sparse projections, so full-image metrics can be harsh.
- Identity can score strongly when source-target frames are visually similar.
- LPIPS was unavailable in this run, so LPIPS metrics are `NaN`.
- GT-depth warp showed outliers and did not behave as a clean upper bound on every metric.
- The notebook reports internal validation results, not a benchmark-scale final result.
- The evaluation supports diagnosis and reporting, but it should not be framed as state-of-the-art NVS performance.

## Recommended Next Steps

- Add pose-direction, intrinsics, and depth-scale calibration for GT-depth warp.
- Add a geometry-valid subset where GT-depth warp beats identity before comparing NVS RGB metrics.
- Use clipped/log-scale plots for AbsRel outlier readability.
- Improve forward splatting or add hole filling before judging full-image RGB quality.
- Compare qualitative examples where adapted improves and regresses relative to DAV2.
- Consider reporting depth metrics and NVS metrics separately.
