# EagleVision Kaggle Notebooks

This folder contains the Kaggle notebooks, ablation notebook, and final report files for the **EagleVision** deep learning research project.

EagleVision investigates whether a pretrained monocular depth model can be improved for RGB-D geometric adaptation using lightweight adapter training, direct metric depth supervision, and carefully weighted geometric consistency objectives.

---

## Main Training Notebook

The main training notebook for the current version of the project is:

```text
eaglevision2.ipynb
```

Use this notebook as the primary entry point for training and evaluating the latest EagleVision pipeline.

> Note: if a local file is named `eglevision2.ipynm`, rename it to `eaglevision2.ipynb` so that GitHub, Jupyter, and Kaggle recognize it correctly as a notebook.

---

## Recommended Kaggle Dataset

Use the following Kaggle dataset as input:

```text
klein2111/scannet-2d
```

Dataset link:

```text
https://www.kaggle.com/datasets/klein2111/scannet-2d
```

### Why this dataset is used

This dataset is used because:

- it is a Kaggle-hosted ScanNet-derived indoor RGB-D dataset
- it is more relevant to indoor paired-view geometry than generic RGB-D alternatives
- it supports reduced-budget training and validation on Kaggle
- it is suitable for testing depth adaptation and geometric consistency objectives

### Dataset caveat

The Kaggle dataset has limited structural documentation and unclear license metadata. The notebooks therefore include dataset inspection and normalization logic before training or evaluation.

If the folder structure differs from the expected ScanNet-style scene layout, update the dataset discovery helper inside the notebook.

---

## Files in This Folder

| File | Role |
|---|---|
| `README.md` | This file. Explains dataset setup, notebook roles, execution order, results, and project justification. |
| `eaglevision2.ipynb` | Main training notebook for the latest EagleVision pipeline. |
| `eaglevision_ablation_report.md` | Final markdown report summarizing the ablation results and project justification. |
| `eaglevision_hyperparameter_ablation_local_cuda_v3_*.ipynb` | Main hyperparameter ablation notebook. Runs reduced-budget experiments and exports numerical results for the paper. |
| `kaggle_phase1_complete.ipynb` | Complete Phase 1 notebook with corrected preprocessing and backbone visualization. |
| `kaggle_phase1_eval_infer.ipynb` | Evaluation and inference notebook for baseline/adapted comparison. |
| `kaggle_phase1_setup_train.ipynb` | Initial setup and training notebook for Kaggle execution. |

---

## Recommended Execution Order

For a clean workflow, run the notebooks in this order.

---

### 1. Initial Setup and Training

```text
kaggle_phase1_setup_train.ipynb
```

Use this notebook to:

- clone the repository inside the Kaggle runtime
- install required dependencies
- locate the Kaggle dataset
- inspect and normalize the dataset structure
- download required checkpoints
- generate configuration files
- run initial training

---

### 2. Evaluation and Inference

```text
kaggle_phase1_eval_infer.ipynb
```

Use this notebook to:

- evaluate the frozen baseline
- evaluate the adapted EagleVision model
- compare validation metrics
- run inference
- inspect qualitative outputs

---

### 3. Complete Phase 1 Pipeline

```text
kaggle_phase1_complete.ipynb
```

Use this notebook for the complete Phase 1 workflow, including:

- corrected input preprocessing
- backbone visualization
- training/evaluation flow
- cleaned outputs for inspection

---

### 4. Main Current Training Pipeline

```text
eaglevision2.ipynb
```

This is the main current training notebook.

Use this notebook for:

- latest EagleVision training
- latest adapter-based depth adaptation experiments
- validation metric generation
- checking the current model behavior

---

### 5. Hyperparameter Ablation

```text
eaglevision_hyperparameter_ablation_local_cuda_v3_*.ipynb
```

Use this notebook for the final ablation study.

It evaluates:

- target-depth loss weight
- cycle-depth loss weight
- combined target/cycle depth-weight grids
- learning rate
- adapter capacity
- RGB reconstruction losses
- camera-motion range

The notebook is designed to produce:

- numerical ablation results
- validation metrics
- exported tables for the paper
- progress logs for long-running cells
- paper-ready markdown summaries

---

### 6. Final Report

```text
eaglevision_ablation_report.md
```

Use this report when writing the final paper section for:

- experimental setup
- main quantitative results
- ablation analysis
- project justification
- limitations
- final claim wording

---

## Kaggle Setup Instructions

### Step 1: Add the dataset

In Kaggle, add the following dataset to the notebook input:

```text
klein2111/scannet-2d
```

### Step 2: Enable GPU

Use a GPU runtime when training or running ablations.

Recommended Kaggle accelerator:

```text
GPU T4 x2
```

A single GPU can also work, but training and ablations will take longer.

### Step 3: Clone the repository

Inside the Kaggle notebook, clone the project repository:

```bash
git clone https://github.com/alooboii/EagleVision.git
cd EagleVision
```

### Step 4: Install dependencies

Run the dependency installation cells included in the notebook.

The notebooks are designed to install missing packages inside the Kaggle runtime.

### Step 5: Run dataset discovery

Before training, run the dataset discovery/preprocessing cells.

These cells inspect the Kaggle input directory and normalize the dataset paths so the training and evaluation code can use them consistently.

---

## Project Objective

The main objective of EagleVision is to test whether a pretrained monocular depth model can be improved for RGB-D geometric adaptation.

The project uses a pretrained depth backbone and adds lightweight trainable components to adapt it using indoor RGB-D geometry. The goal is not to train a depth model from scratch. Instead, the goal is to improve a strong pretrained model using efficient adaptation.

The core research question is:

> Can lightweight adapter-based training with balanced metric depth and geometric consistency objectives improve a pretrained monocular depth model for RGB-D geometric adaptation?

---

## Method Summary

EagleVision is based on the following idea:

1. Start with a pretrained monocular depth model.
2. Keep the pretrained backbone mostly frozen or lightly adapted.
3. Add lightweight trainable adapter components.
4. Train using metric depth supervision and geometric consistency losses.
5. Evaluate whether the adapted model improves over the frozen baseline.

The method is designed to be computationally practical for a course research project while still testing a meaningful deep learning research hypothesis.

---

## Metrics

The main reported metrics are:

| Metric | Meaning | Direction |
|---|---|---|
| AbsRel | Absolute relative depth error | Lower is better |
| RMSE | Root mean squared depth error | Lower is better |
| Depth L1 | Mean absolute depth error | Lower is better |
| Reprojection Depth | Depth/geometric reprojection error | Lower is better |

All reported comparisons use lower-is-better metrics.

---

## Main Ablation Result

The ablation study evaluated **27 configurations**:

- 1 frozen baseline
- 1 default EagleVision configuration
- 25 ablation variants

The most important comparison is shown below.

| Configuration | AbsRel ↓ | RMSE ↓ | Depth L1 ↓ | Reprojection Depth ↓ |
|---|---:|---:|---:|---:|
| Frozen baseline | 0.344097 | 0.851416 | 0.722799 | 0.691047 |
| Default EagleVision | 0.315386 | 0.791003 | 0.662878 | 0.622589 |
| Best overall: `lr_5e-5` | 0.284113 | 0.703733 | 0.580426 | 0.565430 |
| Best AbsRel: `adapter_64` | 0.281092 | 0.719990 | 0.598141 | 0.594139 |

Lower values are better for all metrics.

---

## Improvement Over Frozen Baseline

### Default EagleVision vs. Frozen Baseline

| Metric | Baseline | Default EagleVision | Relative Improvement |
|---|---:|---:|---:|
| AbsRel ↓ | 0.344097 | 0.315386 | 8.34% |
| RMSE ↓ | 0.851416 | 0.791003 | 7.10% |
| Depth L1 ↓ | 0.722799 | 0.662878 | 8.29% |
| Reprojection Depth ↓ | 0.691047 | 0.622589 | 9.91% |

The default EagleVision model improves over the frozen baseline on every primary metric.

### Best Overall Configuration vs. Frozen Baseline

| Metric | Baseline | `lr_5e-5` | Relative Improvement |
|---|---:|---:|---:|
| AbsRel ↓ | 0.344097 | 0.284113 | 17.43% |
| RMSE ↓ | 0.851416 | 0.703733 | 17.35% |
| Depth L1 ↓ | 0.722799 | 0.580426 | 19.70% |
| Reprojection Depth ↓ | 0.691047 | 0.565430 | 18.18% |

The best overall configuration, `lr_5e-5`, gives the strongest balanced improvement across the primary metrics.

---

## Best Configuration

The best configuration by mean metric rank was:

```text
lr_5e-5
```

This configuration achieved:

```text
AbsRel              = 0.284113
RMSE                = 0.703733
Depth L1            = 0.580426
Reprojection Depth  = 0.565430
```

The best configuration by AbsRel alone was:

```text
adapter_64
```

This achieved:

```text
AbsRel = 0.281092
```

However, `lr_5e-5` is the recommended final paper configuration because it gives the strongest balanced performance across multiple metrics, not only one metric.

---

## Cycle-Consistency Interpretation

The cycle-depth ablation shows that cycle consistency must be weighted carefully.

| Configuration | AbsRel ↓ | Interpretation |
|---|---:|---|
| `cycle_depth_0p00` | 0.316103 | Very close to default |
| `cycle_depth_0p25` | 0.314707 | Slightly better than default |
| `default_weights`, cycle depth = 0.35 | 0.315386 | Stable default setting |
| `cycle_depth_1p00` | 0.318206 | Worse than default |
| `cycle_depth_2p00` | 0.320253 | Clearly worse than default |

These results do **not** prove that cycle consistency alone causes the improvement.

The safer conclusion is:

> EagleVision benefits from a balanced combination of pretrained metric depth estimation, lightweight adapter training, direct depth supervision, and carefully weighted geometric consistency.

---

## Project Justification

The results justify the EagleVision project because:

1. The adapted EagleVision model outperforms the frozen pretrained baseline.
2. The best tuned model improves AbsRel by approximately **17.43%**.
3. The best tuned model improves RMSE by approximately **17.35%**.
4. The best tuned model improves Depth L1 by approximately **19.70%**.
5. The best tuned model improves reprojection depth error by approximately **18.18%**.
6. The ablation identifies meaningful trade-offs between learning rate, adapter capacity, and loss weighting.
7. The final configuration is empirically justified rather than arbitrarily selected.

The strongest defensible project claim is:

> A pretrained monocular depth model can be improved for RGB-D geometric adaptation by adding lightweight adapter training and carefully balancing metric depth and geometric consistency objectives.

---

## Recommended Paper Wording

Use this wording in the report or paper:

> Under a fixed reduced-budget validation setting, EagleVision consistently improves over the frozen pretrained baseline. The best overall configuration, `lr_5e-5`, reduces AbsRel from `0.344097` to `0.284113`, RMSE from `0.851416` to `0.703733`, Depth L1 from `0.722799` to `0.580426`, and reprojection depth error from `0.691047` to `0.565430`. These results support the effectiveness of lightweight geometric adaptation. The ablation also shows that cycle-depth consistency must be weighted moderately, since excessive cycle-depth weighting degrades metric depth performance.

---

## Limitations

These experiments should be described as a reduced-budget validation ablation, not as a full benchmark-scale state-of-the-art evaluation.

Main limitations:

- reduced number of scenes
- reduced number of epochs
- relatively small validation split
- no repeated random seeds
- internal validation comparison only
- cycle consistency is not isolated as the sole reason for improvement

Because of these limitations, the final paper should avoid claiming state-of-the-art performance unless a larger benchmark-scale evaluation is added.

---

## Final Conclusion

The ablation study supports the EagleVision project.

Compared with the frozen Depth Anything V2 baseline, the adapted EagleVision model improves all primary depth and geometric metrics. The best overall configuration, `lr_5e-5`, provides the strongest balanced performance across AbsRel, RMSE, Depth L1, and reprojection depth error.

The final contribution should be framed as a combined adaptation pipeline:

> EagleVision improves validation performance over the frozen pretrained baseline by combining pretrained metric depth estimation, lightweight adapter training, direct depth supervision, and carefully weighted geometric consistency.
