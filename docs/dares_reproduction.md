# DARES Reproduction Attempt

This repository now includes a separate DARES path under `src/eaglevision/dares/`.

Implemented pieces:

- Depth Anything V2 backbone from the local baseline implementation.
- Vector-LoRA adapters on selected internal `nn.Linear` modules.
- Decreasing Vector-LoRA capacity from early to late transformer blocks, with explicit per-block overrides.
- SCARED-style stereo manifest loading through the existing endoscopy dataset.
- Self-supervised predicted-depth-to-disparity reprojection.
- Multi-scale SSIM reprojection loss.
- SCARED-style monocular depth metrics and reprojection metrics.

This is intended to match the DARES method family:

- Depth Anything V2 backbone.
- Vector-LoRA, not uniform-rank LoRA.
- SCARED robotic endoscopy stereo protocol.
- Multi-scale SSIM reprojection objective.

Items that must be checked against the paper or released code before reporting official DARES numbers:

- Exact train/validation/test split.
- Exact DA2 checkpoint and encoder variant.
- Exact target modules for Vector-LoRA.
- Exact rank/alpha schedule if specified by released code.
- Exact input resolution and preprocessing.
- Exact optimizer, scheduler, epochs, and augmentations.
- Exact evaluation scaling/alignment rules.

If any of those details differ, label results as a "DARES reproduction attempt," not official DARES numbers.

Commands:

```bash
python -m eaglevision.cli.inspect_dares_modules \
  --config configs/dares/dares_da2s_scared.yaml \
  --contains attn qkv proj fc1 fc2 linear \
  --out outputs/dares/da2s_module_inspection.json
```

```bash
python -m eaglevision.cli.train_dares \
  --config configs/dares/dares_da2s_scared.yaml \
  --train-manifest manifests/scared/train.jsonl \
  --val-manifest manifests/scared/val.jsonl \
  --dry-run
```

```bash
python -m eaglevision.cli.train_dares \
  --config configs/dares/dares_da2s_scared.yaml \
  --train-manifest manifests/scared/train.jsonl \
  --val-manifest manifests/scared/val.jsonl
```

```bash
python -m eaglevision.cli.eval_dares \
  --config configs/dares/dares_da2s_scared.yaml \
  --manifest manifests/scared/test.jsonl \
  --checkpoint outputs/dares/dares_da2s_scared/checkpoints/best.pt \
  --out results/raw/dares_da2s_scared.csv
```

The inspected or guessed target modules are written to `vector_lora_summary.json` during training and should be included with experiment artifacts.
