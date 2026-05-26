from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from eaglevision.losses.endoscopy_cycle_losses import compute_endoscopy_losses
from eaglevision.models.peft_lora_da2 import LoRALinear, apply_lora_to_named_linears, lora_state_dict


class TinyLinearDepth(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.keep = nn.Linear(3, 4)
        self.proj = nn.Linear(3, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pixels = image.permute(0, 2, 3, 1)
        return F.softplus(self.proj(pixels)).squeeze(-1) + 0.5


def test_lora_linear_preserves_shape():
    linear = nn.Linear(5, 7)
    lora = LoRALinear(linear, rank=2, alpha=4)
    x = torch.randn(3, 11, 5)

    y = lora(x)

    assert y.shape == (3, 11, 7)


def test_apply_lora_replaces_only_matching_linears():
    model = TinyLinearDepth()
    summary = apply_lora_to_named_linears(model, ["proj"], rank=2, alpha=4, dropout=0.0)

    assert isinstance(model.proj, LoRALinear)
    assert isinstance(model.keep, nn.Linear)
    assert summary["replaced_modules"] == ["proj"]


def test_only_lora_params_trainable():
    model = TinyLinearDepth()
    apply_lora_to_named_linears(model, ["proj"], rank=2, alpha=4, dropout=0.0)

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]

    assert trainable == ["proj.lora_a.weight", "proj.lora_b.weight"]


def test_lora_training_loss_has_gradients():
    torch.manual_seed(3)
    model = TinyLinearDepth()
    apply_lora_to_named_linears(model, ["proj"], rank=2, alpha=4, dropout=0.0)
    left = torch.rand(1, 3, 5, 7)
    batch = {
        "left_rgb": left,
        "right_rgb": torch.roll(left, shifts=-1, dims=-1),
        "baseline": torch.tensor([1.0]),
        "focal_length": torch.tensor([2.0]),
    }
    depth = model(left)
    log_depth = depth.log()
    outputs = {
        "base_depth": depth,
        "adapted_depth": depth,
        "log_base_depth": log_depth,
        "log_adapted_depth": log_depth,
        "residual": torch.zeros(1, 1, 5, 7),
    }
    config = {
        "weights": {
            "photo": 1.0,
            "cycle_rgb": 0.5,
            "prior_log_l1": 0.0,
            "edge_smoothness": 0.0,
            "supervised_log_l1": 0.0,
            "supervised_silog": 0.0,
            "supervised_disp_l1": 0.0,
            "supervised_disp_log_l1": 0.0,
        },
        "photometric": {"alpha": 0.0},
    }
    loss = compute_endoscopy_losses(batch, outputs, config)["loss_total"]

    loss.backward()

    grad_sum = sum(
        float(parameter.grad.abs().sum().item())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    )
    assert grad_sum > 0.0


def test_lora_state_loads_on_matching_fake_model():
    model = TinyLinearDepth()
    apply_lora_to_named_linears(model, ["proj"], rank=2, alpha=4, dropout=0.0)
    state = lora_state_dict(model)

    reloaded = TinyLinearDepth()
    apply_lora_to_named_linears(reloaded, ["proj"], rank=2, alpha=4, dropout=0.0)
    missing, unexpected = reloaded.load_state_dict(state, strict=False)

    assert not unexpected
    assert "proj.lora_a.weight" not in missing
    assert "proj.lora_b.weight" not in missing
