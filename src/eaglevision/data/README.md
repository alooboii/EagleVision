# Data Layout

Expected ScanNet-style layout:

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
