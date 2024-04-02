from time import perf_counter
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
from typing import Dict, Any
import torch


# RmsPropImplementaiton: https://github.com/hkproj/pytorch-llama
def _norm(x: Tensor, eps: float = 1e-5) -> Tensor:
    """
    Applies Root Mean Square normalization on the input tensor.

    Parameters:
    x (Tensor): Input tensor of shape (Batch, seq_len, Dim).
    eps (float): Small epsilon value to prevent division by zero.

    Returns:
    Tensor: Normalized tensor.
    """
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class RMSNorm(nn.Module):
    """
    Implements Root Mean Square Normalization layer.

    Attributes:
    weight (nn.Parameter): Scaling parameter.
    """

    def __init__(self, feature_dim: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.eps = epsilon
        self.weight = nn.Parameter(torch.ones(feature_dim))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of RMSNorm layer.

        Parameters:
        x (Tensor): Input tensor to normalize.

        Returns:
        Tensor: Normalized and scaled tensor.
        """
        return self.weight * _norm(x, self.eps)


class ConvLayer(nn.Module):
    """
    Regular convolutional layer with optional RMS normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 1,
        bias: bool = True,
        groups: int = 1,
        device: Any = None,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            device=device,
            padding=padding,
            bias=bias,
            groups=groups,
        )

        # for reparameterization
        self.rms_weight = None
        self.module_eval = False

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    @property
    def kernel_size(self):
        return self.conv.kernel_size

    @property
    def padding(self):
        return self.conv.padding

    def load_dict(self, state_dict):
        """
        Loads state dict into the model, with special handling for 'rms_weight'.

        Parameters:
        state_dict (dict): State dictionary with model parameters.
        """
        self.rms_weight = nn.Parameter(
            state_dict.get("rms_weight", torch.ones(self.conv.out_channels))
        )

        filtered_state_dict = {k: v for k, v in state_dict.items() if k != "rms_weight"}
        self.conv.load_state_dict(filtered_state_dict, strict=False)

    def conv_eval(self):
        self.conv = self.conv.eval()

    def eval(self):
        self.module_eval = True

    def forward(self, x: Tensor) -> Tensor:
        # (batch, seq_len, dim) -> (batch, dim, seq_len)
        # to match input shape for nn.Conv1d, which is (batch, dim, length)
        x_transposed = x.transpose(1, 2)
        conv_out = self.conv(x_transposed)

        # performs RMSNorm internally
        if self.module_eval:
            return self.rms_weight * _norm(conv_out.transpose(1, 2))

        # Transpose back to (batch, seq_len, dim)
        return conv_out.transpose(1, 2).type_as(x)


# Fusing conv and RMSNOrm layer. Idea from https://github.com/FrancescoSaverioZuppichini/RepVgg
def get_fused_state_dict(conv: ConvLayer, norm: RMSNorm) -> Dict[str, Tensor]:

    # Remap weights
    dict_ = {"weight": conv.weight, "rms_weight": norm.weight}
    if conv.bias is not None:
        dict_["bias"] = conv.bias

    return dict_


# Inference Version
class RepConvTruncBlock(nn.Module):
    # Truncated version of RepConvBlock. Should be faster.. in therory
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = False,
        use_pre_norm: bool = True,
        groups: int = 1,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        # initial stage
        self.pre_norm = (
            RMSNorm(in_channels, epsilon=eps) if use_pre_norm else nn.Identity()
        )

        # conv layer
        self.k3 = ConvLayer(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=bias,
            groups=groups,
        )

        self.k5 = ConvLayer(
            in_channels,
            out_channels,
            kernel_size=5,
            padding=2,
            bias=bias,
            groups=groups,
        )

        self.k1 = None
        if in_channels != out_channels:
            self.k1 = ConvLayer(
                in_channels, out_channels, kernel_size=1, padding=0, bias=bias
            )

        # final activation stage
        self.rms_comb = RMSNorm(out_channels, epsilon=eps)

        self.act = nn.SiLU(inplace=True)

    def set_eval(self):
        self.k5.conv_eval()
        self.k5.eval()
        self.k3.conv_eval()
        self.k3.eval()

        if self.k1 is not None:
            self.k1.conv_eval()
            self.k1.eval()

        self.act.eval()

    def forward(self, x: Tensor):
        x_norm = self.pre_norm(x)

        # prep main channel for reslike pass
        if self.k1 is not None:
            x_norm = self.k1(x_norm) + self.k3(x_norm) + self.k5(x_norm)
        else:
            # residual connection
            x_norm += self.k3(x_norm) + self.k5(x_norm)

        # normalize before pushing through activation
        x_norm = self.rms_comb(x_norm)

        out = checkpoint(self.act, x_norm, use_reentrant=False)
        return out


class ConvNormActTrunk(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        bias: bool = False,
        groups: int = 1,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.block = ConvLayer(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias,
            groups=groups,
        )

        # self-normalizing so using this
        self.act = nn.SiLU(inplace=True)

    def set_eval(self):
        self.block.conv_eval()
        self.block.eval()

        self.act.eval()

    def forward(self, x: Tensor):
        return self.act(self.block(x))


class ConvNormAct(nn.Module):
    ## need to make repp version of this
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        bias: bool = False,
        groups: int = 1,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.block = ConvLayer(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias,
            groups=groups,
        )

        self.rms = RMSNorm(out_channels, epsilon=eps)

        # self-normalizing so using this
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor):
        x = self.rms(self.block(x))
        x = checkpoint(self.act, x, use_reentrant=False)
        return x

    def to_trunk(self) -> ConvNormActTrunk:
        # intermediate conv layer
        trunked = ConvNormActTrunk(
            self.block.in_channels,
            self.block.out_channels,
            True if self.block.bias is not None else False,
            groups=self.block.groups,
            eps=self.rms_comb.epsilon,
        )

        # Transfer kernel info
        trunked.block.load_dict(get_fused_state_dict(self.bllock, self.rms))

        trunked.act = self.act
        trunked.set_eval()

        return trunked


class RepConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = False,
        use_pre_norm: bool = True,
        groups: int = 1,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.bias = bias
        self.use_pre_norm = use_pre_norm

        # Normalize previous layer
        self.pre_norm = (
            RMSNorm(in_channels, epsilon=eps) if use_pre_norm else nn.Identity()
        )

        # checking immediate neighbours
        self.k3_block = ConvLayer(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=bias,
            groups=groups,
        )
        self.rms_3 = RMSNorm(out_channels, epsilon=eps)

        # checking immediate neighbours
        self.k5_block = ConvLayer(
            in_channels,
            out_channels,
            kernel_size=5,
            padding=2,
            bias=bias,
            groups=groups,
        )
        self.rms_5 = RMSNorm(out_channels, epsilon=eps)

        # Focus on current kernel only
        self.k1_block = None
        if in_channels != out_channels:
            self.k1_block = ConvLayer(
                in_channels, out_channels, kernel_size=1, padding=0, bias=bias
            )

            self.rms_1 = RMSNorm(out_channels, epsilon=eps)

        # activation function
        self.rms_comb = RMSNorm(out_channels)

        # self-normalizing so using this
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor):
        x1 = self.pre_norm(x)

        # local context building
        x3 = self.rms_3(self.k3_block(x1))
        x5 = self.rms_5(self.k5_block(x1))

        # prep main channel for reslike pass
        if self.k1_block is not None:
            x1 = self.rms_1(self.k1_block(x1))

        # residual combination
        out = self.rms_comb(x1 + x3 + x5)

        out = checkpoint(self.act, out, use_reentrant=False)
        return out

    def set_eval(self):
        self.k5_block.conv_eval()
        self.k3_block.conv_eval()

        if self.k1_block is not None:
            self.k1_block.conv_eval()

        self.act.eval()

    def to_trunk(self) -> RepConvTruncBlock:
        # intermediate conv layer
        trunked = RepConvTruncBlock(
            self.k5_block.in_channels,
            self.k5_block.out_channels,
            True if self.k5_block.bias is not None else False,
            groups=self.k5_block.groups,
            eps=self.rms_comb.epsilon,
        )

        # Transfer weights
        if self.use_pre_norm:
            trunked.pre_norm.weight = self.pre_norm.weight

        # Transfer kernel info
        trunked.k5.load_dict(get_fused_state_dict(self.k5_block, self.rms_5))
        trunked.k3.load_dict(get_fused_state_dict(self.k3_block, self.rms_3))

        if self.k1_block is not None:
            trunked.k1.load_dict(get_fused_state_dict(self.k1_block, self.rms_1))

        # Transfer post_rms
        trunked.rms_comb.weight = self.rms_comb.weight
        trunked.set_eval()

        return trunked


def fused_trial():
    channels_in = 6
    channels_out = 6
    sequence_length = 5
    kernel = 3
    padding = 1
    conv_norm = nn.Sequential(
        ConvLayer(
            channels_in, channels_out, kernel_size=kernel, padding=padding, bias=True
        ),
        RMSNorm(channels_out),
    )
    torch.nn.init.uniform_(conv_norm[1].weight)
    with torch.no_grad():
        conv_norm[0].conv_eval()

        conv_fused = ConvLayer(
            conv_norm[0].in_channels,
            conv_norm[0].out_channels,
            kernel_size=conv_norm[0].kernel_size,
            bias=True if conv_norm[0].bias is not None else False,
            padding=conv_norm[0].padding,
        )
        conv_fused.load_dict(get_fused_state_dict(conv_norm[0], conv_norm[1]))

    # input = Batch, Sequence_Len, Dim
    x = torch.randn((1, sequence_length, channels_in))

    start = perf_counter()
    res = conv_norm(x)
    print(f"Final result: \n{res}")
    print(f"Sequential method: {perf_counter() - start:.6f}s")

    # Fusing version
    start = perf_counter()
    conv_fused.eval()
    res1 = conv_fused(x)
    print(f"Final result: \n{res1}")
    print(f"One shot method: {perf_counter() - start:.6f}s")

    assert torch.allclose(res, res1, atol=1e-5)
    print("ALL CLOSEEEEE")


def trial3():
    channels_in = 6
    channels_out = 6
    sequence_length = 13
    bias = False
    fff = RepConvBlock(in_channels=channels_in, out_channels=channels_out, bias=bias)
    fff.set_eval()

    with torch.no_grad():
        x = torch.randn((1, sequence_length, channels_in))
        out_fast = fff.to_trunk()

        start = perf_counter()
        res = fff(x)
        print(f"Training Version: {perf_counter() - start:.6f}s")
        print(f"Print out stuff: \n{res}")
        print("\n===========================\n")
        start = perf_counter()
        res1 = out_fast(x)
        print(f"Truncated Version: {perf_counter() - start:.6f}s")
        print(f"\nReparamed version: \n{res1}")

        assert torch.allclose(res, res1, atol=1e-5)
        print("ALL CLOSEEEEE")


def main():
    torch.manual_seed(0)

    # checking just one block fused
    fused_trial()
    # stacked fused layers
    trial3()


if __name__ == "__main__":
    main()
