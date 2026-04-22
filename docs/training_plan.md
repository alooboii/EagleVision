# Training Plan

Primary Phase 1 mode is `roundtrip_gt_source_depth`.

Training graph:

1. Use source RGB and source GT depth to forward warp into the target camera.
2. Predict target depth from the warped target RGB with the adapted depth model.
3. Fuse valid warped depth with predicted depth using simple hole filling.
4. Backward warp into the source camera.
5. Predict reconstructed-source depth.
6. Optimize target RGB, cycle RGB, cycle depth, and optional target depth anchoring losses.

Default loss weights:

- `target_rgb`: `1.0`
- `cycle_rgb`: `1.0`
- `cycle_depth`: `0.5`
- `target_depth`: `0.25`

The intention is to improve geometric consistency and downstream utility while keeping raw depth quality anchored to real indoor depth.
