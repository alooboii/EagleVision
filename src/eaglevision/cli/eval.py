from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eaglevision.data.collate import scannet_collate
from eaglevision.data.pair_sampler import PairSamplingConfig
from eaglevision.data.scannet_dataset import ScanNetPairDataset
from eaglevision.engine.checkpointing import load_checkpoint
from eaglevision.engine.evaluator import evaluate_model
from eaglevision.models.depth.depth_anything_wrapper import DepthAnythingWithAdapter
from eaglevision.models.rt_depthnvs import RoundTripDepthNVS
from eaglevision.utils.io import load_yaml


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate EagleVision models.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--baseline-only", action="store_true")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    config = load_yaml(args.config)
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Loading evaluation config from {args.config}")
    print(f"Evaluating on device={device}")

    dataset_cfg = config["data"]
    intrinsics = np.array(dataset_cfg["intrinsics"], dtype=np.float32)
    pair_cfg = PairSamplingConfig(**dataset_cfg["pairing"])
    dataset = ScanNetPairDataset(
        root=Path(dataset_cfg["root"]),
        scenes=dataset_cfg["splits"]["val"]["scenes"],
        image_size=tuple(dataset_cfg["image_size"]),
        intrinsics=intrinsics,
        pair_config=pair_cfg,
    )
    print(f"Built evaluation dataset with {len(dataset)} pairs from {len(dataset_cfg['splits']['val']['scenes'])} scenes")
    dataloader = DataLoader(dataset, batch_size=config["eval"]["batch_size"], shuffle=False, collate_fn=scannet_collate)

    depth_cfg = dict(config["model"]["depth"])
    if args.baseline_only:
        depth_cfg["adapter_hidden_channels"] = 1
    depth_model = DepthAnythingWithAdapter(**depth_cfg)
    if args.baseline_only:
        for parameter in depth_model.adapter.parameters():
            parameter.data.zero_()
            parameter.requires_grad = False
    model = RoundTripDepthNVS(depth_model).to(device)
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        load_checkpoint(args.checkpoint, model)
    elif args.baseline_only:
        print("Running baseline-only evaluation with zeroed adapter")
    print("Starting metric computation")
    metrics = evaluate_model(model, dataloader, device, config["losses"]["weights"])
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
