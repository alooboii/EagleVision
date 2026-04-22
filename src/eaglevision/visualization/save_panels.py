from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from eaglevision.utils.io import ensure_dir
from eaglevision.utils.visualization import depth_to_colormap, tensor_to_image


def save_debug_panel(outputs: dict, output_path: Path) -> None:
    """Save a compact multi-panel visualization for one batch item."""
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    panels = [
        ("source_rgb", tensor_to_image(outputs["A"][0])),
        ("source_depth_gt", depth_to_colormap(outputs["D_source_gt"][0])),
        ("target_rgb_gt", tensor_to_image(outputs["B"][0])),
        ("target_warp_rgb", tensor_to_image(outputs["A_t_warp"][0])),
        ("target_warp_depth", depth_to_colormap(outputs["D_t_warp"][0])),
        ("target_pred_depth", depth_to_colormap(outputs["D_t_pred"][0])),
        ("target_fused_depth", depth_to_colormap(outputs["D_t_fused"][0])),
        ("recon_source_rgb", tensor_to_image(outputs["A_s_recon"][0])),
        ("recon_source_depth", depth_to_colormap(outputs["D_s_pred"][0])),
        ("target_holes", outputs["H_t"][0].detach().cpu().numpy()),
    ]
    for axis, (title, image) in zip(axes.flatten(), panels, strict=True):
        axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
