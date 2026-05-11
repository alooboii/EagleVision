# Phase 1 Scope

Phase 1 establishes a reliable geometry-first research backbone for making a pretrained monocular depth estimator more useful for indoor RGB-D geometry.

## In Scope

- ScanNet-style RGB-D scene loading.
- Motion-constrained paired-view sampling.
- Explicit camera intrinsics, poses, projection, backprojection, visibility masks, and geometric warping.
- Depth Anything V2-style baseline integration.
- Lightweight residual depth adapter training.
- Target-depth and round-trip losses.
- Baseline-vs-adapted evaluation.
- Motion bucket, depth-scale, and reprojection diagnostics.
- Kaggle notebooks for reproducible larger runs.
- Markdown reports for ablation and complete evaluation results.

## Out Of Scope

- State-of-the-art novel-view synthesis claims.
- Large-motion photorealistic view synthesis.
- Generative completion of disoccluded regions.
- Learned fusion as the main Phase 1 contribution.
- Full end-to-end source-depth prediction as the primary regime.
- Dataset hosting inside git.
- Large checkpoint hosting in the repository outside intentional Git LFS/release workflows.

## Current Claim Boundary

The project can currently claim that lightweight adapter-based geometric adaptation improves several depth/geometric metrics over the frozen pretrained baseline.

The project should not claim that it fully solves full-frame image synthesis. The complete evaluation report shows that holes, projection coverage, and invalid regions still hurt full-image PSNR/SSIM.

## Success Criteria

Phase 1 is successful when:

- the codebase can train and evaluate the adapted model reproducibly
- baseline and adapted models can be compared on the same pairs
- depth and geometric metrics improve over the frozen baseline
- failure modes are visible through reports, motion buckets, scale diagnostics, and qualitative outputs
- later phases can extend fusion/refinement without replacing the explicit geometry backbone
