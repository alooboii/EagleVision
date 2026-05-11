# Ablation Notebooks

This folder contains the notebooks and markdown report for EagleVision hyperparameter ablations.

## Files

- `kaggle-ablations.ipynb`: main Kaggle ablation notebook.
- `eaglevision_hyperparameter_ablation_local_cuda_v3_unicode_single_progress.ipynb`: local/CUDA ablation variant.
- `eaglevision_ablation_report.md`: final written ablation report.

## Why The Report Lives Here

The ablation report belongs beside the ablation notebooks because it interprets the results produced by those notebooks. Keeping them together makes GitHub navigation clearer and prevents the report from being separated from the experiment that generated it.

## Main Finding

The report concludes that lightweight adapter training improves over the frozen baseline, with the best overall configuration identified as `lr_5e-5` under the reduced-budget validation setup.
