# EagleVision Package

This is the main Python package for EagleVision.

## Modules

- `cli/`: command-line entrypoints.
- `data/`: dataset loading, pair sampling, transforms, and collation.
- `engine/`: training, evaluation, checkpointing, and logging orchestration.
- `losses/`: depth, photometric, and total loss composition.
- `metrics/`: depth, image, and geometric consistency metrics.
- `models/`: adapters, depth wrappers, fusion modules, NVS geometry, and refinement modules.
- `utils/`: geometry, intrinsics, I/O, masks, seeding, and visualization helpers.
- `visualization/`: saveable overlays and panels.

Keep research logic here once it is stable enough to reuse outside a notebook.
