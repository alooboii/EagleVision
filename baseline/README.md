# Baseline

This folder contains the preserved baseline integration used for comparisons against EagleVision.

## Contents

- `depth_anything_v2/`: vendor-style Depth Anything V2 wrapper, inference utilities, checkpoint registry, and CLI.

## Role In The Project

The baseline should remain isolated from new EagleVision research code. New adaptation logic belongs under `src/eaglevision/`; baseline changes should be limited to compatibility fixes, download helpers, or clearly documented inference behavior.

## Commands

```bash
python -m baseline.depth_anything_v2 download
python -m baseline.depth_anything_v2 infer --input <image-or-folder> --output-dir outputs/baseline
```

## Artifact Policy

Checkpoint files should not be committed directly unless intentionally tracked with Git LFS. Keep downloaded baseline weights in local ignored paths or documented external storage.
