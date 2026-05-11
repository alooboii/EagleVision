# Tests

Fast unit tests live here.

## Coverage Areas

- `test_geometry.py`: projection, backprojection, and transforms.
- `test_intrinsics.py`: intrinsics scaling.
- `test_pair_sampler.py`: ScanNet-style pair filtering.
- `test_fusion.py`: depth fusion behavior.
- `test_visibility.py`: visibility/projection masks.
- `test_shapes.py`: model interface shape assumptions.

## Run

```bash
pytest
```

Tests should remain CPU-friendly and suitable for GitHub Actions. Long-running training or notebook validation should be documented separately.
