from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eaglevision.data.collate import scannet_collate
from eaglevision.data.pair_sampler import PairSamplingConfig
from eaglevision.data.scannet_dataset import ScanNetPairDataset
from eaglevision.engine.trainer import Trainer
from eaglevision.models.depth.depth_anything_wrapper import DepthAnythingWithAdapter
from eaglevision.models.rt_depthnvs import RoundTripDepthNVS
from eaglevision.utils.io import ensure_dir, load_yaml
from eaglevision.utils.seed import set_seed


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train EagleVision Phase 1.")
    parser.add_argument("--config", type=Path, required=True)
    return parser


def build_dataset(config: dict, split: str) -> ScanNetPairDataset:
    dataset_cfg = config["data"]
    split_cfg = dataset_cfg["splits"][split]
    pair_cfg = PairSamplingConfig(**dataset_cfg["pairing"])
    intrinsics = np.array(dataset_cfg["intrinsics"], dtype=np.float32)
    return ScanNetPairDataset(
        root=Path(dataset_cfg["root"]),
        scenes=split_cfg["scenes"],
        image_size=tuple(dataset_cfg["image_size"]),
        intrinsics=intrinsics,
        pair_config=pair_cfg,
    )


def main() -> int:
    args = build_argparser().parse_args()
    config = load_yaml(args.config)
    set_seed(int(config["seed"]))
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    train_dataset = build_dataset(config, "train")
    val_dataset = build_dataset(config, "val") if "val" in config["data"]["splits"] else None
    train_loader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=scannet_collate)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=config["eval"]["batch_size"], shuffle=False, collate_fn=scannet_collate)

    depth_model = DepthAnythingWithAdapter(**config["model"]["depth"])
    model = RoundTripDepthNVS(depth_model)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )

    output_dir = ensure_dir(Path(config["output_dir"]))
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=output_dir,
        loss_weights=config["losses"]["weights"],
        log_interval=int(config["train"]["log_interval"]),
        vis_interval=int(config["train"]["vis_interval"]),
        checkpoint_interval=int(config["train"]["checkpoint_interval"]),
    )
    trainer.train(num_epochs=int(config["train"]["epochs"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
