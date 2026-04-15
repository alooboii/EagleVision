from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import trunc_normal_


ENCODER_SETTINGS = {
    "vits": {
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "features": 64,
        "out_channels": [48, 96, 192, 384],
        "intermediate_layers": [2, 5, 8, 11],
    },
    "vitb": {
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "features": 128,
        "out_channels": [96, 192, 384, 768],
        "intermediate_layers": [2, 5, 8, 11],
    },
    "vitl": {
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
        "intermediate_layers": [4, 11, 17, 23],
    },
}

METRIC_MAX_DEPTH = {
    "hypersim": 20.0,
    "vkitti": 80.0,
}


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_values, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.img_size = (img_size, img_size)
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, h, w = x.shape
        patch_h, patch_w = self.patch_size
        if h % patch_h != 0 or w % patch_w != 0:
            raise ValueError(
                f"Input shape {(h, w)} must be divisible by patch size {(patch_h, patch_w)}"
            )

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFNFused(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        hidden = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        self.w12 = nn.Linear(in_features, hidden * 2)
        self.w3 = nn.Linear(hidden, out_features)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class MemEffAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        bsz, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, tokens, 3, self.num_heads, dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = q @ k.transpose(-2, -1)
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(bsz, tokens, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class NestedTensorBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: float | None = 1.0,
        drop_path_rate: float = 0.0,
        ffn_layer: str = "mlp",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MemEffAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            proj_bias=True,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        if ffn_layer == "swiglufused":
            self.mlp = SwiGLUFFNFused(dim, hidden_dim, dim)
        else:
            self.mlp = Mlp(dim, hidden_dim, dim, act_layer=nn.GELU, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path_rate

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        init_values: float = 1.0,
        ffn_layer: str = "mlp",
        interpolate_offset: float = 0.1,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.interpolate_offset = interpolate_offset
        self.num_register_tokens = 0

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.register_tokens = None

        self.blocks = nn.ModuleList(
            [
                NestedTensorBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                    init_values=init_values,
                    ffn_layer=ffn_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self._init_weights()

    def _init_weights(self) -> None:
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.mask_token, std=1e-6)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def interpolate_pos_encoding(self, x: Tensor, width: int, height: int) -> Tensor:
        npatch = x.shape[1] - 1
        n_ref = self.pos_embed.shape[1] - 1

        if npatch == n_ref and width == height:
            return self.pos_embed

        pos_embed = self.pos_embed.float()
        class_pos = pos_embed[:, :1]
        patch_pos = pos_embed[:, 1:]

        width_tokens = width // self.patch_size
        height_tokens = height // self.patch_size

        width_tokens = width_tokens + self.interpolate_offset
        height_tokens = height_tokens + self.interpolate_offset

        sqrt_n = math.sqrt(n_ref)
        scale_factor = (float(width_tokens) / sqrt_n, float(height_tokens) / sqrt_n)

        patch_pos = patch_pos.reshape(1, int(sqrt_n), int(sqrt_n), self.embed_dim)
        patch_pos = patch_pos.permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, scale_factor=scale_factor, mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, self.embed_dim)

        return torch.cat([class_pos, patch_pos], dim=1).to(x.dtype)

    def prepare_tokens_with_masks(self, x: Tensor) -> Tensor:
        _, _, width, height = x.shape
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, width, height)
        return x

    def get_intermediate_layers(
        self,
        x: Tensor,
        n: int | Sequence[int],
        return_class_token: bool = False,
        norm: bool = True,
    ) -> tuple[Tensor, ...] | tuple[tuple[Tensor, Tensor], ...]:
        x = self.prepare_tokens_with_masks(x)
        outputs: list[Tensor] = []

        if isinstance(n, int):
            blocks_to_take = list(range(len(self.blocks) - n, len(self.blocks)))
        else:
            blocks_to_take = list(n)

        block_set = set(blocks_to_take)
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in block_set:
                outputs.append(x)

        if norm:
            outputs = [self.norm(out) for out in outputs]

        class_tokens = [out[:, 0] for out in outputs]
        patch_tokens = [out[:, 1 + self.num_register_tokens :] for out in outputs]

        if return_class_token:
            return tuple(zip(patch_tokens, class_tokens))

        return tuple(patch_tokens)


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int, activation: nn.Module, use_bn: bool) -> None:
        super().__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(features) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(features) if use_bn else nn.Identity()
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    def __init__(
        self,
        features: int,
        activation: nn.Module,
        use_bn: bool,
        align_corners: bool = True,
        size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.align_corners = align_corners
        self.size = size

        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, bias=True)
        self.resConfUnit1 = ResidualConvUnit(features, activation, use_bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, use_bn)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs: Tensor, size: tuple[int, int] | None = None) -> Tensor:
        output = xs[0]
        if len(xs) == 2:
            output = self.skip_add.add(output, self.resConfUnit1(xs[1]))

        output = self.resConfUnit2(output)

        if size is not None:
            resize_kwargs = {"size": size}
        elif self.size is not None:
            resize_kwargs = {"size": self.size}
        else:
            resize_kwargs = {"scale_factor": 2}

        output = F.interpolate(output, mode="bilinear", align_corners=self.align_corners, **resize_kwargs)
        return self.out_conv(output)


def _make_scratch(in_shape: list[int], out_shape: int) -> nn.Module:
    scratch = nn.Module()
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
    return scratch


class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        features: int,
        out_channels: list[int],
        use_bn: bool,
        use_clstoken: bool,
        output_activation: str,
    ) -> None:
        super().__init__()

        self.use_clstoken = use_clstoken
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(in_channels=in_channels, out_channels=out_channel, kernel_size=1, stride=1, padding=0)
                for out_channel in out_channels
            ]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        if use_clstoken:
            self.readout_projects = nn.ModuleList(
                [nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU()) for _ in range(len(self.projects))]
            )

        self.scratch = _make_scratch(out_channels, features)
        self.scratch.stem_transpose = None

        make_fusion = partial(FeatureFusionBlock, features, nn.ReLU(False), use_bn, True)
        self.scratch.refinenet1 = make_fusion()
        self.scratch.refinenet2 = make_fusion()
        self.scratch.refinenet3 = make_fusion()
        self.scratch.refinenet4 = make_fusion()

        self.scratch.output_conv1 = nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1)

        if output_activation == "sigmoid":
            final_activation: nn.Module = nn.Sigmoid()
        elif output_activation == "relu":
            final_activation = nn.ReLU(True)
        else:
            raise ValueError("output_activation must be 'relu' or 'sigmoid'")

        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            final_activation,
            nn.Identity(),
        )

    def forward(self, out_features: tuple[tuple[Tensor, Tensor], ...], patch_h: int, patch_w: int) -> Tensor:
        decoded = []

        for i, feature in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = feature
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), dim=-1))
            else:
                x = feature[0]

            x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            decoded.append(x)

        layer1, layer2, layer3, layer4 = decoded

        layer1_rn = self.scratch.layer1_rn(layer1)
        layer2_rn = self.scratch.layer2_rn(layer2)
        layer3_rn = self.scratch.layer3_rn(layer3)
        layer4_rn = self.scratch.layer4_rn(layer4)

        path4 = self.scratch.refinenet4(layer4_rn, size=layer3_rn.shape[2:])
        path3 = self.scratch.refinenet3(path4, layer3_rn, size=layer2_rn.shape[2:])
        path2 = self.scratch.refinenet2(path3, layer2_rn, size=layer1_rn.shape[2:])
        path1 = self.scratch.refinenet1(path2, layer1_rn)

        out = self.scratch.output_conv1(path1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        return self.scratch.output_conv2(out)


class DepthAnythingV2(nn.Module):
    def __init__(
        self,
        encoder: str,
        features: int,
        out_channels: list[int],
        use_bn: bool = False,
        use_clstoken: bool = False,
        max_depth: float | None = None,
    ) -> None:
        super().__init__()

        if encoder not in ENCODER_SETTINGS:
            raise ValueError(f"Unsupported encoder '{encoder}'")

        self.encoder = encoder
        settings = ENCODER_SETTINGS[encoder]

        self.intermediate_layer_idx = {
            "vits": ENCODER_SETTINGS["vits"]["intermediate_layers"],
            "vitb": ENCODER_SETTINGS["vitb"]["intermediate_layers"],
            "vitl": ENCODER_SETTINGS["vitl"]["intermediate_layers"],
        }

        ffn_layer = "swiglufused" if encoder == "vitg" else "mlp"
        self.pretrained = DinoVisionTransformer(
            img_size=518,
            patch_size=14,
            embed_dim=settings["embed_dim"],
            depth=settings["depth"],
            num_heads=settings["num_heads"],
            init_values=1.0,
            ffn_layer=ffn_layer,
            interpolate_offset=0.1,
        )

        output_activation = "sigmoid" if max_depth is not None else "relu"
        self.depth_head = DPTHead(
            in_channels=settings["embed_dim"],
            features=features,
            out_channels=out_channels,
            use_bn=use_bn,
            use_clstoken=use_clstoken,
            output_activation=output_activation,
        )
        self.max_depth = max_depth

    def forward(self, x: Tensor) -> Tensor:
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.pretrained.get_intermediate_layers(
            x,
            self.intermediate_layer_idx[self.encoder],
            return_class_token=True,
        )
        depth = self.depth_head(features, patch_h, patch_w)

        if self.max_depth is None:
            depth = F.relu(depth)
        else:
            depth = depth * self.max_depth

        return depth.squeeze(1)


def create_model(mode: str, encoder: str, profile: str = "hypersim") -> DepthAnythingV2:
    if encoder not in ENCODER_SETTINGS:
        raise ValueError(f"Unsupported encoder '{encoder}'. Expected one of: {', '.join(ENCODER_SETTINGS)}")

    settings = ENCODER_SETTINGS[encoder]
    max_depth = METRIC_MAX_DEPTH[profile] if mode == "metric" else None

    return DepthAnythingV2(
        encoder=encoder,
        features=settings["features"],
        out_channels=list(settings["out_channels"]),
        use_bn=False,
        use_clstoken=False,
        max_depth=max_depth,
    )
