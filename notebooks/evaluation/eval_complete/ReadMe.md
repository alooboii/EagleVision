# EagleVision Complete Evaluation Report

This folder contains the final result-inspection notebooks and a report summary generated from `eval-complete-final.ipynb`.

## Source Notebook

- [`eval-complete-final.ipynb`](eval-complete-final.ipynb): complete evaluation dashboard copied here for direct review beside this report.
- [`eval.ipynb`](eval.ipynb): earlier evaluation result notebook.
- [`check.ipynb`](check.ipynb): result checking notebook.

The report below is based on the final dashboard output in `eval-complete-final.ipynb`.

## Evaluation Setup

| Item | Value |
|---|---|
| Runtime | Kaggle GPU evaluation |
| Device | CUDA |
| Dataset | ScanNet-style `klein2111/scannet-2d` |
| Dataset discovery | 1,513 ScanNet-style scenes found |
| Validation scenes | 152 scenes |
| Evaluation pairs | 433 source/target pairs |
| Image size | `192 x 288` |
| Depth mode | Metric |
| Encoder | `vits` |
| Baseline checkpoint | Depth Anything V2 metric Hypersim `vits` |
| Adapted checkpoint | `best_100ep.pt`, step `293590` |
| LPIPS | Unavailable, reported as `NaN` |
| SSIM | Available through `skimage` |

## Methods Compared

| Method | Description |
|---|---|
| `Identity baseline` | Source image used as a no-warp image baseline. |
| `GT-depth warp` | Geometry warp using ground-truth depth. |
| `DAV2 baseline` | Frozen Depth Anything V2 metric baseline. |
| `Adapted 100ep` | EagleVision adapted model after the 100-epoch run. |

## Main Comparison

Lower is better for `abs_rel`, `rmse`, `hole_ratio`, `reprojection_depth_l1`, `rgb_l1_valid`, and `median_scaled_absrel`. Higher is better for `delta1`, `psnr_*`, `ssim_*`, and `valid_pixel_ratio`.

| Method | AbsRel ↓ | RMSE ↓ | Delta1 ↑ | PSNR Full ↑ | PSNR Valid ↑ | SSIM Full ↑ | SSIM Valid ↑ | Valid Pixel Ratio ↑ | Hole Ratio ↓ | Reprojection Depth L1 ↓ | RGB L1 Valid ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Adapted 100ep | 0.344948 | 2.187455 | 0.395220 | 8.030335 | 11.299679 | 0.079778 | 0.748439 | 0.403856 | 0.596144 | 2.020791 | 0.221797 |
| DAV2 baseline | 0.461546 | 2.371236 | 0.312454 | 8.472656 | 11.418896 | 0.102726 | 0.704900 | 0.467533 | 0.532467 | 2.208435 | 0.216783 |
| GT-depth warp | 2.300415 | 2.620927 | 0.563551 | 7.986739 | 11.555642 | 0.062490 | 0.782498 | 0.368082 | 0.631918 | 2.345473 | 0.215017 |
| Identity baseline | NaN | NaN | NaN | 13.679151 | 13.679151 | 0.399479 | 0.399479 | 1.000000 | 0.000000 | NaN | 0.145775 |

## Adapted 100ep vs DAV2 Baseline

The adapted model improves the main depth/geometric metrics, but it regresses some full-image image-quality and coverage metrics.

| Metric | Mean Improvement | Median Improvement | Adapted Better On | Pairs | Paired t-test p | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
| AbsRel | 0.114954 | 0.048052 | 63.92% | 413 | 1.56e-15 | 5.20e-16 |
| RMSE | 0.143124 | 0.073363 | 60.53% | 413 | 7.77e-08 | 1.65e-11 |
| Delta1 | 0.082628 | 0.036248 | 60.05% | 413 | 2.53e-07 | 8.10e-07 |
| SSIM valid | 0.039255 | 0.040862 | 74.64% | 414 | 9.08e-19 | 8.98e-22 |
| Reprojection depth L1 | 0.149793 | 0.086295 | 62.95% | 413 | 4.80e-08 | 8.41e-13 |
| PSNR full | -0.442322 | -0.259539 | 29.56% | 433 | 1.24e-27 | 2.71e-26 |
| PSNR valid | -0.135256 | -0.055729 | 47.10% | 414 | 8.35e-02 | 1.29e-01 |
| SSIM full | -0.022948 | -0.012608 | 27.25% | 433 | 5.35e-22 | 3.38e-26 |
| Valid pixel ratio | -0.063676 | -0.066388 | 27.71% | 433 | 2.66e-22 | 2.00e-22 |
| Hole ratio | -0.063676 | -0.066388 | 27.71% | 433 | 2.66e-22 | 2.01e-22 |
| RGB L1 valid | -0.005261 | -0.000631 | 47.83% | 414 | 4.17e-02 | 4.44e-02 |

## Motion Bucket Results

| Method | Motion | AbsRel ↓ | RMSE ↓ | Delta1 ↑ | PSNR Valid ↑ | SSIM Valid ↑ | Valid Pixel Ratio ↑ | Hole Ratio ↓ | Reprojection Depth L1 ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Adapted 100ep | easy | 0.368480 | 1.037587 | 0.193571 | 18.422159 | 0.701899 | 0.821349 | 0.178651 | 0.891225 |
| DAV2 baseline | easy | 0.428597 | 1.210460 | 0.305194 | 17.741586 | 0.666432 | 0.809371 | 0.190629 | 1.047396 |
| GT-depth warp | easy | 0.058200 | 0.226099 | 0.963522 | 19.861383 | 0.746491 | 0.799003 | 0.200997 | 0.119763 |
| Adapted 100ep | medium | 0.349500 | 2.914390 | 0.398997 | 11.466657 | 0.730525 | 0.451527 | 0.548473 | 2.720482 |
| DAV2 baseline | medium | 0.479213 | 3.091698 | 0.293025 | 11.646166 | 0.688538 | 0.512834 | 0.487166 | 2.904515 |
| GT-depth warp | medium | 3.278319 | 3.578792 | 0.601864 | 11.710375 | 0.765573 | 0.399553 | 0.600447 | 3.224967 |
| Adapted 100ep | hard | 0.335070 | 0.716440 | 0.391833 | 10.804794 | 0.786044 | 0.301521 | 0.698479 | 0.605403 |
| DAV2 baseline | hard | 0.423983 | 0.835176 | 0.354766 | 10.779985 | 0.741280 | 0.371418 | 0.628582 | 0.725270 |
| GT-depth warp | hard | 0.350833 | 0.715612 | 0.476563 | 11.065984 | 0.817361 | 0.297309 | 0.702691 | 0.596629 |

## Depth Scale Diagnostics

| Method | Raw AbsRel ↓ | Median-Scaled AbsRel ↓ | Scale Ratio | Scale Bias Mean | Scale Bias Median |
|---|---:|---:|---:|---:|---:|
| Adapted 100ep | 0.344948 | 0.196013 | 2.483105 | 1.021666 | 0.989214 |
| DAV2 baseline | 0.461546 | 0.184480 | 2.491017 | 1.157676 | 1.121515 |
| GT-depth warp | 2.300415 | 0.200742 | 1.008610 | 3.117333 | 3.040386 |
| Identity baseline | NaN | NaN | NaN | NaN | NaN |

Interpretation: the adapted model reduces raw absolute relative depth error compared with DAV2, but the median-scaled metric slightly favors the baseline. This suggests that some of the adapted model's gain comes from better metric scale/bias rather than uniformly better scale-normalized local structure.

## Runtime And Memory

| Method | Avg Inference Time / Pair | Pairs / Second | Peak CUDA Memory | Adapter Overhead vs DAV2 |
|---|---:|---:|---:|---:|
| Adapted 100ep | 0.053536 s | 18.678848 | 228.318848 MB | 0.772807% |
| DAV2 baseline | 0.053126 s | 18.823199 | 228.318848 MB | NaN |
| GT-depth warp | 0.030316 s | 32.985463 | 228.318848 MB | NaN |
| Identity baseline | 0.021861 s | 45.744534 | 228.318848 MB | NaN |

The adapter adds less than 1% inference overhead compared with the frozen DAV2 baseline in this run.

## Skipped Or Failed Rows

| Method | Skipped | Count |
|---|---:|---:|
| Adapted 100ep | false | 433 |
| DAV2 baseline | false | 433 |
| GT-depth warp | false | 433 |
| Identity baseline | false | 433 |

No methods were skipped for the evaluated pairs.

## Final Interpretation

The adapted 100-epoch EagleVision model improves 4 out of 5 main paired metrics against the DAV2 baseline, so the result is positive for depth/geometric adaptation but mixed overall.

Strong improvements:

- AbsRel improves by `0.114954` on average and is better on `63.92%` of paired examples.
- RMSE improves by `0.143124` on average and is better on `60.53%` of paired examples.
- Delta1 improves by `0.082628` on average and is better on `60.05%` of paired examples.
- Valid-region SSIM improves by `0.039255` on average and is better on `74.64%` of paired examples.
- Reprojection depth L1 improves by `0.149793` on average and is better on `62.95%` of paired examples.

Regressions and caveats:

- Full-image PSNR and SSIM regress, likely because full-image scores penalize holes and invalid projected regions.
- Adapted average hole ratio is `0.596144`, higher than the DAV2 baseline's `0.532467`.
- Valid pixel ratio drops from `0.467533` for DAV2 to `0.403856` for the adapted model.
- Valid-region RGB L1 slightly regresses.
- The notebook reported a sanity-ranking warning for `rgb_l1_valid`, `psnr_valid`, and `abs_rel`; this means pose convention, depth scale, and intrinsics should be reviewed before overclaiming image-reconstruction conclusions.

Overall conclusion:

> The adapted EagleVision model gives a meaningful improvement in direct depth and reprojection-depth metrics over the frozen DAV2 baseline, with minimal runtime overhead. The result is not a clean win on full-image reconstruction metrics because hole coverage and invalid regions worsen. The strongest defensible claim is that the 100-epoch adapted model improves depth/geometric accuracy, while full-frame image synthesis quality remains limited by visibility coverage, holes, and geometry setup.
