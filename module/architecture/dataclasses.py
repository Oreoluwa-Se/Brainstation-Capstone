from dataclasses import dataclass
from typing import Optional


@dataclass
class ConvInputs:
    in_channels: int = -1
    out_channels: int = -1
    kernel_size: int = 1
    bias: bool = True
    groups: int = 1
    dilation_rate: int = 1
    eps: float = 1e-5
    act_out: bool = True
    along_sequence: bool = False
    device = None


@dataclass
class ModelArgs:
    dim: int = -1
    n_layers: int = -1
    n_heads: int = -1
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    attn_dropout_rate: float = 0.1
    proj_dropout_rate: float = 0.5
    batch_size: int = 32
    seq_len: int = 2048
    num_target_classes: int = 2
    use_amp: bool = False
    downsample = False
    device: str = None


@dataclass
class EGArgs:
    in_channels: int = -1
    out_channels: int = -1
    kernel_size: int = -1
    dilation_rate: int = -1
    groups: int = -1
