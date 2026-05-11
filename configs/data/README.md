# Data Configs

Dataset configuration files live here.

## Current Files

- `scannet.yaml`: ScanNet-style RGB-D scene layout configuration.

## Expected Dataset Shape

```text
scene_id/
  color/
  depth/
  pose/
```

Keep actual datasets out of git. Point configs at local, Kaggle, or mounted dataset roots.
