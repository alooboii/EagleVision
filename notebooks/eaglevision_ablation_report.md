# EagleVision Ablation Report

## Project Title

**EagleVision: Lightweight Geometric Adaptation of a Pretrained Monocular Depth Model**

---

## 1. Executive Summary

This report summarizes the final results from the EagleVision hyperparameter ablation study. The main purpose of the experiment was to determine whether the proposed EagleVision adaptation pipeline improves over a frozen pretrained Depth Anything V2 baseline, and whether the final training configuration can be justified empirically for the research paper.

The results support the project direction. The adapted EagleVision model improves over the frozen baseline across the primary depth and geometric metrics. The best overall configuration, `lr_5e-5`, gives a substantial improvement over the frozen baseline:

- **AbsRel improves by approximately 17.43%**
- **RMSE improves by approximately 17.35%**
- **Depth L1 improves by approximately 19.70%**
- **Reprojection Depth improves by approximately 18.18%**

Therefore, the results justify the core claim that lightweight adapter-based training with balanced depth and geometric objectives improves validation performance over directly using the frozen pretrained model.

However, the cycle-consistency results should be interpreted carefully. The results do not prove that cycle consistency alone is the main reason for improvement. Instead, they show that moderate cycle-depth weighting can be useful, while excessive cycle-depth weighting degrades performance.

---

## 2. Experimental Goal

The ablation study was designed to answer the following questions:

1. Does EagleVision improve over the frozen pretrained Depth Anything V2 baseline?
2. Which hyperparameter settings produce the best validation performance?
3. Are the chosen loss weights empirically justified?
4. Does cycle-depth consistency improve performance, or does it introduce a trade-off?
5. Can the final configuration be defended in the research paper?

The study was conducted as a reduced-budget validation ablation. This means that the experiment was intentionally smaller than a full benchmark-scale evaluation, but it was still useful for comparing configurations under a consistent validation setup.

---

## 3. Ablation Setup

A total of **27 configurations** were evaluated.

These included:

- **1 frozen baseline**
  - `baseline_only`

- **1 default EagleVision configuration**
  - `default_weights`

- **25 ablation variants** covering:
  - target-depth loss weight
  - cycle-depth loss weight
  - combined depth-weight grids
  - learning rate
  - adapter capacity
  - RGB reconstruction losses
  - camera-motion range

All configurations were evaluated using the same fixed validation split. This makes the comparisons internally consistent.

---

## 4. Main Quantitative Results

Lower values are better for all metrics in the table below.

| Configuration | AbsRel ↓ | RMSE ↓ | Depth L1 ↓ | Reprojection Depth ↓ |
|---|---:|---:|---:|---:|
| Frozen baseline | 0.344097 | 0.851416 | 0.722799 | 0.691047 |
| Default EagleVision | 0.315386 | 0.791003 | 0.662878 | 0.622589 |
| Best overall rank: `lr_5e-5` | 0.284113 | 0.703733 | 0.580426 | 0.565430 |
| Best AbsRel: `adapter_64` | 0.281092 | 0.719990 | 0.598141 | 0.594139 |

The results show that the default EagleVision configuration already improves over the frozen baseline. The tuned configurations improve performance further.

---

## 5. Improvement Over Frozen Baseline

### 5.1 Default EagleVision vs. Frozen Baseline

| Metric | Baseline | Default EagleVision | Relative Improvement |
|---|---:|---:|---:|
| AbsRel ↓ | 0.344097 | 0.315386 | 8.34% |
| RMSE ↓ | 0.851416 | 0.791003 | 7.10% |
| Depth L1 ↓ | 0.722799 | 0.662878 | 8.29% |
| Reprojection Depth ↓ | 0.691047 | 0.622589 | 9.91% |

The default EagleVision model improves all four primary metrics compared with the frozen pretrained baseline. This shows that adaptation is useful even before additional hyperparameter tuning.

### 5.2 Best Overall Configuration vs. Frozen Baseline

| Metric | Baseline | `lr_5e-5` | Relative Improvement |
|---|---:|---:|---:|
| AbsRel ↓ | 0.344097 | 0.284113 | 17.43% |
| RMSE ↓ | 0.851416 | 0.703733 | 17.35% |
| Depth L1 ↓ | 0.722799 | 0.580426 | 19.70% |
| Reprojection Depth ↓ | 0.691047 | 0.565430 | 18.18% |

The best overall configuration, `lr_5e-5`, gives a much stronger improvement over the frozen baseline. This supports the conclusion that the default learning rate was conservative and that the adapter benefits from a higher learning rate under the reduced training budget.

---

## 6. Best Configuration

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

This is the strongest final configuration if the goal is balanced performance across multiple depth and geometric metrics.

The best single configuration by AbsRel was:

```text
adapter_64
```

This achieved:

```text
AbsRel = 0.281092
```

However, `lr_5e-5` is the better final choice for the paper because it performs best overall across the evaluated metrics, not only on one metric.

---

## 7. Interpretation of Ablation Results

The ablation results support three main findings.

### 7.1 Adapter Training Improves Performance

The adapted EagleVision model performs better than the frozen pretrained Depth Anything V2 baseline. This means that the project is not only relying on the pretrained model. The lightweight adaptation stage contributes measurable improvement.

### 7.2 Hyperparameter Selection Matters

The learning-rate ablation produced the strongest overall configuration. Increasing the learning rate to `5e-5` improved the ability of the adapter to learn useful domain-specific geometric corrections within the reduced training budget.

### 7.3 Adapter Capacity Affects Depth Accuracy

The `adapter_64` configuration achieved the best AbsRel score. This suggests that the smaller adapter may limit the model's ability to adapt fully. A larger adapter gives the model more capacity to learn useful refinements.

---

## 8. Cycle-Consistency Analysis

The cycle-depth ablation gives a more nuanced result.

| Configuration | AbsRel ↓ | Interpretation |
|---|---:|---|
| `cycle_depth_0p00` | 0.316103 | Very close to default |
| `cycle_depth_0p25` | 0.314707 | Slightly better than default |
| `default_weights`, cycle depth = 0.35 | 0.315386 | Stable default setting |
| `cycle_depth_1p00` | 0.318206 | Worse than default |
| `cycle_depth_2p00` | 0.320253 | Clearly worse than default |

These results show that cycle-depth consistency must be weighted carefully. A small or moderate cycle-depth weight is stable and can slightly improve results, but a high cycle-depth weight hurts performance.

Therefore, the correct interpretation is:

> EagleVision benefits from a balanced combination of direct metric depth supervision, geometric consistency, and adapter-based fine-tuning. Cycle consistency is useful as a regularizing signal, but it is not the only reason the method works.

The paper should avoid claiming that cycle consistency alone causes the improvement.

---

## 9. Does This Justify the Project?

Yes, the results justify the project.

The project is justified because:

1. The adapted EagleVision model outperforms the frozen pretrained baseline.
2. The best tuned model improves AbsRel by approximately **17.43%**.
3. The best tuned model improves RMSE by approximately **17.35%**.
4. The best tuned model improves Depth L1 by approximately **19.70%**.
5. The best tuned model improves reprojection depth error by approximately **18.18%**.
6. The ablation identifies meaningful trade-offs between learning rate, adapter capacity, and loss-weight choices.
7. The results provide an empirical basis for choosing the final configuration instead of selecting hyperparameters arbitrarily.

The strongest defensible claim is:

> A pretrained monocular depth model can be improved for RGB-D geometric adaptation by adding lightweight adapter training and carefully balancing metric depth and geometric consistency objectives.

---

## 10. Limitations

These results should be presented as a reduced-budget validation ablation, not as a full state-of-the-art benchmark evaluation.

Important limitations include:

- The ablation used a reduced number of scenes.
- Training was run for a reduced number of epochs.
- The validation split was relatively small.
- Results were not averaged across multiple random seeds.
- The experiment mainly supports internal comparison between configurations.
- The experiment does not prove that cycle consistency alone is responsible for the improvement.

Because of these limitations, the paper should use careful wording and avoid overstating the result.

---

## 11. Recommended Paper Wording

A safe and accurate way to describe the result is:

> Under a fixed reduced-budget validation setting, EagleVision consistently improves over the frozen pretrained baseline. The best overall configuration, `lr_5e-5`, reduces AbsRel from `0.344097` to `0.284113`, RMSE from `0.851416` to `0.703733`, Depth L1 from `0.722799` to `0.580426`, and reprojection depth error from `0.691047` to `0.565430`. These results support the effectiveness of lightweight geometric adaptation. The ablation also shows that cycle-depth consistency must be weighted moderately, since excessive cycle-depth weighting degrades metric depth performance.

---

## 12. Final Conclusion

The ablation study supports the EagleVision project. Compared with the frozen Depth Anything V2 baseline, the adapted EagleVision model improves all primary depth and geometric metrics. The best overall configuration, `lr_5e-5`, achieves the strongest balanced performance across AbsRel, RMSE, Depth L1, and reprojection depth error.

The results justify the use of lightweight adapter-based training and balanced geometric objectives. However, the contribution should be framed as a combined adaptation pipeline rather than as proof that cycle consistency alone is responsible for the improvement.

Final conclusion:

> EagleVision improves validation performance over the frozen pretrained baseline by combining pretrained metric depth estimation, lightweight adapter training, direct depth supervision, and carefully weighted geometric consistency. The ablation study provides empirical justification for the final configuration and supports the project as a valid research direction.
