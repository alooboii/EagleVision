# Data

Dataset loading, pair sampling, transforms, and batch collation live here.

## Files

- `scannet_dataset.py`: ScanNet-style RGB-D scene dataset.
- `pair_sampler.py`: candidate camera-pair filtering and motion thresholds.
- `transforms.py`: image/depth preprocessing transforms.
- `collate.py`: batch collation helpers.

## Expected Layout

The default data path expects a ScanNet-style layout:

```text
data/scannet/
  scene0000_00/
    color/
      0.jpg
    depth/
      0.png
    pose/
      0.txt
```

Depth is assumed to be meters, or millimeters that can be converted by dividing by `1000`.

## Policy

Actual datasets should not be committed. Keep datasets in local `data/`, Kaggle inputs, or mounted storage and point YAML configs at the dataset root.
