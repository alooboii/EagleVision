# Scripts

This folder contains thin shell or Python wrappers around package entrypoints.

## Files

- `prepare_scannet.py`: helper for preparing ScanNet-style data.
- `run_train.sh`: training wrapper.
- `run_eval.sh`: evaluation wrapper.
- `run_compare_baseline_vs_adapted.sh`: comparison wrapper.
- `run_demo.sh`: demo round-trip wrapper.

## Rule

Keep reusable logic in `src/eaglevision/`. Scripts should stay small and should mainly compose existing package CLIs.
