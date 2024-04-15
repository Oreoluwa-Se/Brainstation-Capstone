from dataclasses import dataclass
from time import perf_counter
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
from typing import Dict, Any
import torch
import torch.nn.functional as F
from .dataclasses import ConvInputs


# --------------------- RMS NORM IMPLEMENTATION ---------------------
# RmsPropImplementaiton: https://github.com/hkproj/pytorch-llama
def _norm(x: Tensor, eps: float = 1e-5, axis: int = -1) -> Tensor:
    """
    Applies Root Mean Square normalization on the input tensor.

    Parameters:
    x (Tensor): Input tensor.
    eps (float): Small epsilon value to prevent division by zero.
    axis (int): The dimension to normalize over.

    Returns:
    Tensor: Normalized tensor.
    """
    mean_sq = x.pow(2).mean(dim=axis, keepdim=True)
    return x * torch.rsqrt(mean_sq + eps)


class RMSNorm(nn.Module):
    """
    Implements Root Mean Square Normalization layer.

    Attributes:
    weight (nn.Parameter): Scaling parameter.
    """

    def __init__(self, feature_dim: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.eps = epsilon
        self.feature_dim = feature_dim
        self.weight = nn.Parameter(torch.ones(feature_dim))

    def forward(self, x: Tensor, axis: int = -1) -> Tensor:
        """
        Forward pass of RMSNorm layer.

        Parameters:
        x (Tensor): Input tensor to normalize.
        axis (int): The axis to perform the normalization.

        Returns:
        Tensor: Normalized and scaled tensor.
        """
        # Normalize the input tensor along the specified axis
        normalized = _norm(x, self.eps, axis)

        # Adjust the weight to match the dimension being normalized
        shape = [1] * len(x.shape)
        shape[axis] = self.feature_dim
        weight = self.weight.view(shape)

        return weight * normalized.type_as(x)


def build_conv(args: ConvInputs) -> nn.Conv1d:
    return nn.Conv1d(
        args.in_channels,
        args.out_channels,
        args.kernel_size,
        padding=(args.kernel_size - 1) // 2 * args.dilation_rate,
        dilation=args.dilation_rate,
        groups=args.groups,
    )


# --------------------- GATED DILATED MODULE ---------------------
class GatedDilatedModule(nn.Module):
    def __init__(self, args: ConvInputs, pre_norm: bool = False) -> None:
        super().__init__()

        self.along_sequence = args.along_sequence
        out_channels = args.out_channels
        self.act_out = args.act_out

        # gating used for activation
        self.gate_branch = nn.Sequential(
            nn.Linear(args.in_channels, args.in_channels), nn.SiLU()
        )

        self.pre_norm = (
            RMSNorm(feature_dim=args.in_channels, epsilon=args.eps)
            if pre_norm
            else None
        )

        # dilated convs
        args.out_channels = args.in_channels
        args.groups = args.in_channels
        args.dilation_rate = 1
        self.d_1 = build_conv(args)

        args.dilation_rate = 2
        self.d_2 = build_conv(args)

        args.dilation_rate = 4
        self.d_4 = build_conv(args)

        # normalization layer
        self.comb_norm = RMSNorm(feature_dim=args.in_channels, epsilon=args.eps)

        # Final layer - reintroduce out channels
        args.kernel_size = 1
        args.out_channels = out_channels
        args.groups = 1
        args.dilation_rate = 1
        self.fin_norm = RMSNorm(feature_dim=args.in_channels, epsilon=args.eps)
        self.fin_layer = nn.Sequential(build_conv(args), nn.SiLU())

    def _gate_out(self, x: Tensor):
        act = self.gate_branch(x.transpose(1, 2))
        return act.transpose(1, 2)

    def forward(self, x: Tensor):
        # batch, seq, dim
        x_h = x.transpose(1, 2) if not self.along_sequence else x

        if self.pre_norm:
            x_h = self.pre_norm.forward(x_h, axis=1)

        # dilated context build
        x_d = self.d_1(x_h) + self.d_2(x_h) + self.d_4(x_h)

        # same scale addition
        x_h = x_h + self.comb_norm.forward(x_d * self._gate_out(x_h), 1)

        # activate final layer
        x_h = self.fin_norm(x_h, axis=1)
        x_h = F.silu(self.fin_layer(x_h)) if self.act_out else self.fin_layer(x_h)

        return x_h.transpose(1, 2) if not self.along_sequence else x_h


# --------------------- CONV NORM ACT MODULE ---------------------
class ConvNormAct(nn.Module):
    def __init__(self, args: ConvInputs, pre_norm=False) -> None:
        super().__init__()

        self.along_sequence = args.along_sequence
        out_channels = args.out_channels
        self.act_out = args.act_out
        self.channel_boost = args.kernel_size > 1

        self.pre_norm = (
            RMSNorm(feature_dim=args.in_channels, epsilon=args.eps)
            if pre_norm
            else None
        )

        # gating used for activation
        if self.channel_boost:
            self.gate_branch = nn.Sequential(
                nn.Linear(args.in_channels, args.in_channels), nn.SiLU()
            )

            # dilated convs
            args.out_channels = args.in_channels
            args.groups = args.in_channels
            args.dilation_rate = 1
            self.d_1 = build_conv(args)

            # normalization layer
            self.comb_norm = RMSNorm(feature_dim=args.in_channels, epsilon=args.eps)

        # Final layer - reintroduce out channels
        args.kernel_size = 1
        args.out_channels = out_channels
        args.groups = 1
        args.dilation_rate = 1
        self.fin_norm = RMSNorm(feature_dim=args.in_channels, epsilon=args.eps)
        self.fin_layer = nn.Sequential(build_conv(args), nn.SiLU())

    def _gate_out(self, x: Tensor):
        act = self.gate_branch(x.transpose(1, 2))
        return act.transpose(1, 2)

    def forward(self, x: Tensor):
        # batch, seq, dim
        x_h = x.transpose(1, 2) if not self.along_sequence else x

        if self.pre_norm:
            x_h = self.pre_norm.forward(x_h, axis=1)

        if self.channel_boost:
            # same scale addition
            x_h = x_h + self.comb_norm.forward(self.d_1(x_h) * self._gate_out(x_h), 1)

        # activate final layer
        x_h = self.fin_norm(x_h, axis=1)
        x_h = F.silu(self.fin_layer(x_h)) if self.act_out else self.fin_layer(x_h)

        return x_h.transpose(1, 2) if not self.along_sequence else x_h
