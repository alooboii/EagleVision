# Experiments

Supported Phase 1 experiment modes:

- `roundtrip_gt_source_depth`: main training mode using GT source depth for the forward warp.
- `baseline_depth_eval`: evaluate the frozen baseline model on the same splits and metrics.
- `adapted_depth_eval`: evaluate the trained adapted model on depth and geometric buckets.
- `roundtrip_pred_source_depth`: structurally supported follow-on path for later phases.

Recommended comparisons:

1. Baseline depth-only model versus adapted model on target-depth metrics.
2. Baseline depth-only model versus adapted model inside the same round-trip evaluation loop.
3. Ablation on motion thresholds, target-depth supervision weight, and adapter capacity.
