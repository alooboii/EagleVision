from __future__ import annotations

import torch
from torch import nn

from eaglevision.dares.vector_lora import (
    VectorLoRALinear,
    apply_vector_lora,
    build_rank_schedule,
    list_trainable_parameters,
)


class TinyBlocks(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict({"attn": nn.ModuleDict({"qkv": nn.Linear(4, 4), "proj": nn.Linear(4, 4)}), "other": nn.Linear(4, 4)}),
                nn.ModuleDict({"attn": nn.ModuleDict({"qkv": nn.Linear(4, 4), "proj": nn.Linear(4, 4)}), "other": nn.Linear(4, 4)}),
            ]
        )


def test_vector_lora_linear_shape_and_frozen_base():
    linear = nn.Linear(5, 7)
    wrapped = VectorLoRALinear(linear, rank=2, alpha=4, module_name="blocks.0.attn.qkv", block_index=0)
    out = wrapped(torch.randn(3, 11, 5))
    assert out.shape == (3, 11, 7)
    assert not wrapped.linear.weight.requires_grad
    assert wrapped.lora_A.weight.requires_grad
    assert wrapped.lora_B.weight.requires_grad


def test_decreasing_schedule_monotonic_and_explicit_override():
    config = {"early_rank": 16, "middle_rank": 8, "late_rank": 4, "final_rank": 2}
    schedule = build_rank_schedule(12, config)
    ranks = [schedule[index] for index in range(12)]
    assert ranks == sorted(ranks, reverse=True)
    explicit = build_rank_schedule(3, {"per_block_rank": {0: 9, 1: 5, 2: 1}})
    assert explicit == {0: 9, 1: 5, 2: 1}


def test_apply_vector_lora_replaces_matching_linears_and_gradients_only_lora():
    model = TinyBlocks()
    summary = apply_vector_lora(model, ["qkv", "proj"], {"early_rank": 4, "final_rank": 2, "default_rank": 2}, dropout=0.0)
    assert len(summary["replaced_modules"]) == 4
    trainable = list_trainable_parameters(model)
    assert trainable
    assert all(".lora_A." in name or ".lora_B." in name for name in trainable)

    x = torch.randn(2, 4)
    out = model.blocks[0]["attn"]["qkv"](x).sum()
    out.backward()
    assert model.blocks[0]["attn"]["qkv"].lora_B.weight.grad is not None
    assert model.blocks[0]["attn"]["qkv"].linear.weight.grad is None
