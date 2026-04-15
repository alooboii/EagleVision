# EagleVision

This repository includes a baseline implementation of **Depth Anything V2** under:

`baseline/depth_anything_v2`

The CLI entrypoint is:

```bash
python -m baseline.depth_anything_v2 <command>
```

## Environment

Depth Anything V2 baseline targets Python **3.11 or 3.12**.

Install dependencies with your preferred tool, for example:

```bash
pip install -e .
```

## Commands

### 1) Download checkpoints

Downloads official checkpoints into `baseline/depth_anything_v2/checkpoints` by default.

```bash
python -m baseline.depth_anything_v2 download
```

Examples:

```bash
# relative-only small and base
python -m baseline.depth_anything_v2 download --mode relative --encoder vits --encoder vitb

# metric-only (indoor/hypersim profile)
python -m baseline.depth_anything_v2 download --mode metric --profile hypersim

# metric-only (outdoor/vkitti profile)
python -m baseline.depth_anything_v2 download --mode metric --profile vkitti
```

### 2) Run inference

Supports both a single image path and a directory path.

```bash
python -m baseline.depth_anything_v2 infer \
  --input <image-or-folder> \
  --output-dir <output-folder>
```

Examples:

```bash
# relative depth with vitl (default mode/encoder)
python -m baseline.depth_anything_v2 infer \
  --input assets/example.jpg \
  --output-dir outputs/relative

# metric depth with indoor profile (hypersim)
python -m baseline.depth_anything_v2 infer \
  --input assets/example.jpg \
  --output-dir outputs/metric_hypersim \
  --mode metric \
  --profile hypersim \
  --encoder vitl

# metric depth with outdoor profile (vkitti)
python -m baseline.depth_anything_v2 infer \
  --input assets/example.jpg \
  --output-dir outputs/metric_vkitti \
  --mode metric \
  --profile vkitti \
  --encoder vitb
```

You can also pass a specific checkpoint path directly:

```bash
python -m baseline.depth_anything_v2 infer \
  --input assets/example.jpg \
  --output-dir outputs/custom_ckpt \
  --checkpoint /path/to/checkpoint.pth
```

## Output contract

For each input image, the command writes:

- `<stem>_depth.npy` (raw float32 depth map)
- `<stem>_depth.png` (normalized colorized preview)

When the input is a directory, relative subfolders are preserved in the output directory.
