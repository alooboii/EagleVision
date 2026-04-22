# Architecture

Phase 1 treats the depth estimator as the main object of improvement. The repository wraps the vendor baseline `baseline/depth_anything_v2` with a frozen backbone plus a small residual adapter, then places that model inside an explicit geometry-first round-trip rendering loop.

Core modules:

- `data/`: ScanNet-style paired-view loading and motion-constrained pair sampling.
- `models/nvs/`: explicit backprojection, transform, reprojection, visibility, and z-buffer rasterization.
- `models/depth/`: baseline Depth Anything V2 wrapper plus light trainable adaptation.
- `models/rt_depthnvs.py`: forward warp, target-depth prediction, fusion, backward warp, and reconstructed-source depth prediction.
- `losses/` and `metrics/`: masked geometric and depth objectives aligned to the Phase 1 framing.

The rendering loop is intentionally interpretable and modular so later phases can add learned fusion, refinement, or generative completion without replacing the geometric backbone.
