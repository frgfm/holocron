# Copyright (C) 2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn
from torchvision.ops import StochasticDepth

from holocron.nn import GlobalAvgPool2d

from ..checkpoints import Checkpoint, _handle_legacy_pretrained
from ..utils import _configure_model, conv_sequence
from ._features import _ClassifierMixin

__all__ = ["IFormer", "iformer_m", "iformer_s", "iformer_t"]


def _conv_norm(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 1,
    stride: int = 1,
    padding: int = 0,
    groups: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        *conv_sequence(
            in_channels,
            out_channels,
            norm_layer=nn.BatchNorm2d,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )
    )


class _Residual(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        channels: int,
        stochastic_depth_prob: float,
        layer_scale_init_value: float,
    ) -> None:
        super().__init__()
        self.block = block
        self.drop_path = StochasticDepth(stochastic_depth_prob, "row")
        self.layer_scale = (
            nn.Parameter(torch.full((1, channels, 1, 1), layer_scale_init_value))
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.drop_path(self.block(x))
        if self.layer_scale is not None:
            residual = self.layer_scale * residual
        return x + residual


class _ConvBlock(_Residual):
    def __init__(
        self,
        channels: int,
        ratio: int,
        stochastic_depth_prob: float,
        layer_scale_init_value: float,
    ) -> None:
        hidden_channels = ratio * channels
        super().__init__(
            nn.Sequential(
                _conv_norm(channels, channels, 7, padding=3, groups=channels),
                _conv_norm(channels, hidden_channels),
                nn.GELU(),
                _conv_norm(hidden_channels, channels),
            ),
            channels,
            stochastic_depth_prob,
            layer_scale_init_value,
        )


class _RepCPE(_Residual):
    def __init__(self, channels: int, stochastic_depth_prob: float, layer_scale_init_value: float) -> None:
        super().__init__(
            _conv_norm(channels, channels, 3, padding=1, groups=channels),
            channels,
            stochastic_depth_prob,
            layer_scale_init_value,
        )


class _SelfModulationAttention(nn.Module):
    def __init__(self, channels: int, ratio: int, head_dim_reduce_ratio: int) -> None:
        super().__init__()
        hidden_channels = ratio * channels
        attention_channels = channels // head_dim_reduce_ratio
        self.scale = attention_channels**-0.5
        self.q = _conv_norm(channels, attention_channels)
        self.k = _conv_norm(channels, attention_channels)
        self.v_gate = _conv_norm(channels, 2 * hidden_channels)
        self.proj = _conv_norm(hidden_channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, _, height, width = x.shape
        value, gate = self.v_gate(x).sigmoid().chunk(2, dim=1)
        query = self.q(x).flatten(2) * self.scale
        key = self.k(x).flatten(2)
        attention = (query.transpose(-2, -1) @ key).softmax(dim=-1)
        value = value.flatten(2) @ attention.transpose(-2, -1)
        return self.proj(value.reshape(batch_size, -1, height, width) * gate)


class _AttentionBlock(_Residual):
    def __init__(
        self,
        channels: int,
        ratio: int,
        head_dim_reduce_ratio: int,
        stochastic_depth_prob: float,
        layer_scale_init_value: float,
    ) -> None:
        super().__init__(
            _SelfModulationAttention(channels, ratio, head_dim_reduce_ratio),
            channels,
            stochastic_depth_prob,
            layer_scale_init_value,
        )


class _FFN(_Residual):
    def __init__(
        self,
        channels: int,
        ratio: int,
        stochastic_depth_prob: float,
        layer_scale_init_value: float,
    ) -> None:
        hidden_channels = ratio * channels
        super().__init__(
            nn.Sequential(
                _conv_norm(channels, hidden_channels),
                nn.GELU(),
                _conv_norm(hidden_channels, channels),
            ),
            channels,
            stochastic_depth_prob,
            layer_scale_init_value,
        )


class _FusedIB(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        hidden_channels = 4 * in_channels
        super().__init__(
            _conv_norm(
                in_channels,
                hidden_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            ),
            nn.GELU(),
            _conv_norm(hidden_channels, out_channels),
        )


class IFormer(_ClassifierMixin, nn.Sequential):
    """Implements iFormer as described in
    ["iFormer: Integrating ConvNet and Transformer for Mobile Application"](https://arxiv.org/abs/2501.15369).

    Args:
        channels: number of output channels in each stage
        stage_layouts: per-stage counts of leading convolution blocks, attention triplets, and trailing convolutions
        conv_ratio: expansion ratio of convolution blocks
        ffn_ratio: expansion ratio of feed-forward blocks
        num_classes: number of output classes
        in_channels: number of input channels
        stochastic_depth_prob: maximum stochastic-depth probability
        layer_scale_init_value: initial residual layer-scale value; disabled when zero
        act_layer: activation used by the fused inverted-bottleneck stem
    """

    def __init__(
        self,
        channels: list[int],
        stage_layouts: list[tuple[int, int, int]],
        conv_ratio: int,
        ffn_ratio: int,
        num_classes: int = 10,
        in_channels: int = 3,
        stochastic_depth_prob: float = 0.0,
        layer_scale_init_value: float = 0.0,
        act_layer: Callable[[], nn.Module] = nn.GELU,
    ) -> None:
        if len(channels) != 4 or len(stage_layouts) != 4:
            raise ValueError("`channels` and `stage_layouts` are expected to contain four stages")

        num_blocks = sum(num_conv + 3 * num_attention + num_tail for num_conv, num_attention, num_tail in stage_layouts)
        drop_rates = torch.linspace(0, stochastic_depth_prob, num_blocks).tolist()
        block_idx = 0
        features: list[nn.Module] = [
            nn.Sequential(
                _conv_norm(in_channels, channels[0] // 2, 5, stride=2, padding=2),
                act_layer(),
                _FusedIB(channels[0] // 2, channels[0], 5, stride=2),
            )
        ]
        for stage_idx, (stage_channels, layout) in enumerate(zip(channels, stage_layouts, strict=True)):
            if stage_idx:
                features.append(_conv_norm(channels[stage_idx - 1], stage_channels, 3, stride=2, padding=1))

            num_conv, num_attention, num_tail = layout
            stage: list[nn.Module] = []
            for _ in range(num_conv):
                stage.append(
                    _ConvBlock(
                        stage_channels,
                        conv_ratio,
                        drop_rates[block_idx],
                        layer_scale_init_value,
                    )
                )
                block_idx += 1
            for _ in range(num_attention):
                stage.extend([
                    _RepCPE(stage_channels, drop_rates[block_idx], layer_scale_init_value),
                    _AttentionBlock(
                        stage_channels,
                        1,
                        2 if stage_idx == 2 else 4,
                        drop_rates[block_idx + 1],
                        layer_scale_init_value,
                    ),
                    _FFN(
                        stage_channels,
                        ffn_ratio,
                        drop_rates[block_idx + 2],
                        layer_scale_init_value,
                    ),
                ])
                block_idx += 3
            for _ in range(num_tail):
                stage.append(
                    _ConvBlock(
                        stage_channels,
                        conv_ratio,
                        drop_rates[block_idx],
                        layer_scale_init_value,
                    )
                )
                block_idx += 1
            features.append(nn.Sequential(*stage))

        head = nn.Sequential(nn.BatchNorm1d(channels[-1]), nn.Linear(channels[-1], num_classes))
        super().__init__(
            OrderedDict([
                ("features", nn.Sequential(*features)),
                ("pool", GlobalAvgPool2d(flatten=True)),
                ("head", head),
            ])
        )

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def _iformer(
    checkpoint: Checkpoint | None,
    progress: bool,
    channels: list[int],
    stage_layouts: list[tuple[int, int, int]],
    conv_ratio: int,
    ffn_ratio: int,
    **kwargs: Any,
) -> IFormer:
    model = IFormer(channels, stage_layouts, conv_ratio, ffn_ratio, **kwargs)
    return _configure_model(model, checkpoint, progress=progress)


def iformer_t(
    pretrained: bool = False,
    checkpoint: Checkpoint | None = None,
    progress: bool = True,
    **kwargs: Any,
) -> IFormer:
    """iFormer-T model.

    Args:
        pretrained: If True, loads the default checkpoint when one is available
        checkpoint: If specified, sets the model parameters to the checkpoint values
        progress: If True, displays a download progress bar
        kwargs: keyword arguments of [`IFormer`][holocron.models.classification.iformer.IFormer]

    Returns:
        An iFormer-T model
    """
    checkpoint = _handle_legacy_pretrained(pretrained, checkpoint, None)
    return _iformer(
        checkpoint, progress, [32, 64, 128, 256], [(2, 0, 0), (2, 0, 0), (6, 3, 1), (0, 2, 0)], 3, 2, **kwargs
    )


def iformer_s(
    pretrained: bool = False,
    checkpoint: Checkpoint | None = None,
    progress: bool = True,
    **kwargs: Any,
) -> IFormer:
    """iFormer-S model.

    Args:
        pretrained: If True, loads the default checkpoint when one is available
        checkpoint: If specified, sets the model parameters to the checkpoint values
        progress: If True, displays a download progress bar
        kwargs: keyword arguments of [`IFormer`][holocron.models.classification.iformer.IFormer]

    Returns:
        An iFormer-S model
    """
    checkpoint = _handle_legacy_pretrained(pretrained, checkpoint, None)
    return _iformer(
        checkpoint, progress, [32, 64, 176, 320], [(2, 0, 0), (2, 0, 0), (9, 3, 1), (0, 2, 0)], 4, 3, **kwargs
    )


def iformer_m(
    pretrained: bool = False,
    checkpoint: Checkpoint | None = None,
    progress: bool = True,
    **kwargs: Any,
) -> IFormer:
    """iFormer-M model.

    Args:
        pretrained: If True, loads the default checkpoint when one is available
        checkpoint: If specified, sets the model parameters to the checkpoint values
        progress: If True, displays a download progress bar
        kwargs: keyword arguments of [`IFormer`][holocron.models.classification.iformer.IFormer]

    Returns:
        An iFormer-M model
    """
    checkpoint = _handle_legacy_pretrained(pretrained, checkpoint, None)
    return _iformer(
        checkpoint, progress, [48, 96, 192, 384], [(2, 0, 0), (2, 0, 0), (9, 4, 1), (0, 2, 0)], 4, 3, **kwargs
    )
