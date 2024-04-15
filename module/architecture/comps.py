from ..preprocess.load_and_batch import MetaData
from .base import RMSNorm

from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torch
from typing import List, Dict, Optional
from .dataclasses import ModelArgs, EGArgs, ConvInputs
from module.architecture.base import GatedDilatedModule, ConvNormAct, RMSNorm


def make_divisible(n, div):
    return n + div - (n % div) if n % div != 0 else n


def precompute_rope_theta(
    head_dim: int, seq_len: int, device: str, theta: float = 10000.0
):
    # Build the theta parameters
    assert head_dim % 2 == 0, "Head dimension must be divisible by 2"

    # shape: (head_dim / 2)
    theta_num = torch.arange(0, head_dim, 2).float()
    # shape: (head_dim / 2)
    theta = 1.0 / (theta ** (theta_num / head_dim)).to(device)
    # shape: (seq_len)
    m = torch.arange(seq_len, device=device)
    # shape: (seq_len, Head_dim/2)
    freqs = torch.outer(m, theta).float()
    # we use complex form for compact way of storing sin-cos
    # Shape: (seq_len, head_dim/2) -> (seq_len, head_dim/2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embeddings(x: Tensor, freqs_complex: Tensor, device: str):
    # (B, seq_len, h, head_dim) -> (B, seq_len, h, head_dim/2)
    x_complex = torch.view_as_complex(
        x.float().reshape(*x.shape[:-1], -1, 2).contiguous()
    )

    # (seq_len, head_dim/2) -> (1, seq_len, 1, head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (B, seq_len, h, head_dim/2) * (1, seq_len, 1, head_dim/2) -> (B, seq_len, h, head_dim/2)
    x_rotated = x_complex * freqs_complex
    # (B, seq_len, h, head_dim/2) -> (B, seq_len, h, head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, seq_len, h, head_dim/2, 2) -> (B, seq_len, h, head_dim)
    x_out = x_out.reshape(*x.shape)

    return x_out.type_as(x).to(device)


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    if n_rep == 1:
        return x

    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]  # (B, Seq_len, n_kv_heads,1, head_dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs, inter_channel: int = -1) -> None:
        super().__init__()

        dim = inter_channel if inter_channel != -1 else args.dim
        hidden_dim = 2 * dim
        if args.ffn_dim_multiplier is not None and args.ffn_dim_multiplier > 0:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)

        # round the hidden_dim to nearest multiple of parameters
        hidden_dim = args.multiple_of * (
            (hidden_dim + args.multiple_of - 1) // args.multiple_of
        )

        # for swiglu calculation
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: Tensor):
        swish = F.silu(self.w1(x))
        x_v = self.w3(x)
        x = swish * x_v

        # activated layer
        return self.w2(x)


class GlobalContext(nn.Module):
    """
    Peforms global aggregation using a linear layers then uses convolutions for the q,k,v weights
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_pre_norm: bool = False,
        eps: float = 1e-5,
    ) -> None:

        super().__init__()

        # Normalize previous layer
        self.pre_norm = (
            RMSNorm(in_channels, epsilon=eps) if use_pre_norm else nn.Identity()
        )

        self.gc = nn.Linear(in_channels, out_channels, bias=True)
        self.norm = RMSNorm(out_channels, epsilon=eps)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 2)
        # shrink the channel space.
        x_act = self.pre_norm(x)
        x_act = self.act(self.gc(x_act))
        x_dropout = self.dropout(x_act)

        return x_dropout.transpose(1, 2)


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.head_dim = args.dim // args.n_heads
        self.sqrt_head_dim = self.head_dim**0.5
        self.use_auto_cast = args.use_amp and torch.cuda.is_available()

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads

        # ratio between number of heads for query and kv
        self.n_rep = self.n_heads_q // self.n_kv_heads

        # dimensions
        self.query_dim = int(args.n_heads * self.head_dim)
        self.kv_dim = int(self.n_kv_heads * self.head_dim)

        # local qkv - global qkv
        total_dim = self.query_dim + 2 * self.kv_dim
        self.lc_weights = nn.Linear(args.dim, total_dim, bias=False)
        self.gc_weights = nn.Linear(args.dim, total_dim, bias=False)

        # normalizations for query and key [pre attention]
        self.q_lc_norm = RMSNorm(self.query_dim, epsilon=args.norm_eps)
        self.k_lc_norm = RMSNorm(self.kv_dim, epsilon=args.norm_eps)

        self.q_gc_norm = RMSNorm(self.query_dim, epsilon=args.norm_eps)
        self.k_gc_norm = RMSNorm(self.kv_dim, epsilon=args.norm_eps)

        # attention dropout rate
        self.dropout = nn.Dropout(args.attn_dropout_rate)

        # linear projections for local and global outputs
        self.local_out = nn.Linear(2 * self.query_dim, self.query_dim)
        self.local_proj_dropout = nn.Dropout(args.proj_dropout_rate)
        self.lc_out_norm = RMSNorm(self.query_dim, epsilon=args.norm_eps)

        self.global_out = nn.Linear(2 * self.query_dim, self.query_dim)
        self.global_proj_dropout = nn.Dropout(args.proj_dropout_rate)
        self.gc_out_norm = RMSNorm(self.query_dim, epsilon=args.norm_eps)

    def _context_build(
        self, x: Tensor, weight_func: nn.Module, q_norm: RMSNorm, k_norm: RMSNorm
    ):
        n_batch, seq_len, _ = x.shape

        comb = weight_func(x)
        dims = [self.query_dim, self.kv_dim, self.kv_dim]
        query, key, value = torch.split(comb, dims, dim=-1)

        # (B, seq_len, dim) -> (B, seq_len, h_q, q_dim)
        query = q_norm(query).view(n_batch, seq_len, self.n_heads_q, -1)

        # (B, seq_len, dim) -> (B, seq_len, h_kv, kv_dim)
        key = k_norm(key).view(n_batch, seq_len, self.n_kv_heads, -1)
        value = value.view(n_batch, seq_len, self.n_kv_heads, -1)

        return query, key, value

    def _compute_attention(self, q_lc, k_lc, v_lc, q_gc, k_gc, v_gc, mask):
        # (B, seq_len, h_q, q_dim) -> (B, seq_len, h_q, Dim)
        xq = torch.cat([q_lc, q_gc], dim=-1)
        # (B, seq_len, h_kv, kv_dim) -> (B, seq_len, h_kv, Dim)
        xk = torch.cat([k_lc, k_gc], dim=-1)
        xv = torch.cat([v_lc, v_gc], dim=-1)

        # if doing the grouped query version.. might need to repeat heads
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # ------> Now perform Multihead Attn <----------
        # (B, seq_len, head, Dim) -> (B, head, seq_len, Dim)
        xq, xk, xv = [tmp.transpose(1, 2) for tmp in (xq, xk, xv)]

        # (B, head, seq_len, head_dim)
        return checkpoint(self._attention_forward, xq, xk, xv, use_reentrant=False)

    def _attention_forward(self, q, k, v, mask=None):
        # (B, head, seq_len, Dim) @ (B, head, Dim, seq_len) -> (B, head, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.sqrt_head_dim
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        return torch.matmul(attn, v)

    def forward(
        self, local_c: Tensor, global_c: Tensor, freqs: Tensor, mask: Tensor = None
    ):
        batch_size, seq_len, _ = local_c.shape
        device = local_c.device

        with autocast(enabled=self.use_auto_cast):
            # ------> QKV Local <----------
            q_lc, k_lc, v_lc = self._context_build(
                local_c, self.lc_weights, self.q_lc_norm, self.k_lc_norm
            )

            # ------> QKV Global <----------
            q_gc, k_gc, v_gc = self._context_build(
                global_c, self.gc_weights, self.q_gc_norm, self.k_gc_norm
            )

            if freqs is not None:
                # including position encodings
                q_lc = apply_rotary_embeddings(q_lc, freqs, device)
                k_lc = apply_rotary_embeddings(k_lc, freqs, device)

                q_gc = apply_rotary_embeddings(q_gc, freqs, device)
                k_gc = apply_rotary_embeddings(k_gc, freqs, device)
            # ------> Merge local and global Information <----------
            # (B, head, seq_len, head_dim)
            out = self._compute_attention(q_lc, k_lc, v_lc, q_gc, k_gc, v_gc, mask)

        # passing through final layers
        # (B, head, seq_len, dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Mixing the activated layers
        x_lc = self.lc_out_norm(self.local_out(out))
        x_lc = self.local_proj_dropout(x_lc)

        x_gc = self.gc_out_norm(self.global_out(out))
        x_gc = self.global_proj_dropout(x_gc)

        # returning normalized version
        return x_lc, x_gc


class EncoderBlock(nn.Module):
    def __init__(self, args=ModelArgs, dwnsample: bool = False) -> None:
        super().__init__()

        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads

        self.pre_norm = RMSNorm(args.dim, epsilon=args.norm_eps)

        self.lc = GatedDilatedModule(self._get_conv_args(args), pre_norm=True)
        self.gc = GlobalContext(
            args.seq_len, args.seq_len, use_pre_norm=True, eps=args.norm_eps
        )

        self.lc_norm = RMSNorm(args.dim, epsilon=args.norm_eps)
        self.gc_norm = RMSNorm(args.dim, epsilon=args.norm_eps)

        self.attn = SelfAttention(args)

        self.lc_ffn_norm = RMSNorm(args.dim, epsilon=args.norm_eps)
        self.gc_ffn_norm = RMSNorm(args.dim, epsilon=args.norm_eps)

        self.ff_lc = FeedForward(args)
        self.ff_gc = FeedForward(args)

        self.fin_layer = self._fin_layer(args)

        self.dwnsample = None
        if dwnsample:
            dnwsample_args = self._get_conv_args(args)
            # reduce the sequence length
            dnwsample_args.out_channels = dnwsample_args.in_channels // 2
            self.dwnsample = GatedDilatedModule(dnwsample_args, pre_norm=True)

    def _fin_layer(self, args: ModelArgs):
        fin_args = self._get_conv_args(args)

        fin_args.in_channels = 2 * args.dim
        fin_args.out_channels = args.dim
        fin_args.along_sequence = False
        fin_args.kernel_size = 1

        return ConvNormAct(fin_args, pre_norm=True)

        # qkv extraction

    def _get_conv_args(self, args: ModelArgs):

        return ConvInputs(
            in_channels=args.seq_len,
            out_channels=args.seq_len,
            kernel_size=3,
            bias=True,
            groups=1,
            dilation_rate=1,
            eps=args.norm_eps,
            act_out=True,
            along_sequence=True,
        )

    def forward(self, x: Tensor, freqs: Tensor, mask: Tensor = None):
        # build local context
        lc = self.lc(x)
        lc_norm = self.lc_norm(lc)

        # build global context
        gc = self.gc(x)
        gc_norm = self.gc_norm(gc)

        lca_norm, gca_norm = self.attn.forward(lc_norm, gc_norm, freqs, mask)

        # lc and gc here are logits
        lc = self.ff_lc(self.lc_ffn_norm(lc_norm + lca_norm))
        gc = self.ff_gc(self.gc_ffn_norm(gc_norm + gca_norm))

        out = x + self.fin_layer(torch.cat([lc, gc], dim=-1))

        if self.dwnsample:
            return self.dwnsample(out)

        return out


class EmbeddingGenerator(nn.Module):
    def __init__(self, seq_len: int, eps: float = 1e-5):
        super(EmbeddingGenerator, self).__init__()
        # pre norm
        self.pre_norm = RMSNorm(seq_len, eps)

        # gating branch
        self.gate_branch = nn.Sequential(nn.Linear(seq_len, seq_len), nn.SiLU())

        # Dilated Learning
        args = EGArgs(seq_len, seq_len, 3, 1, seq_len)
        self.conv = self._build_conv(args)

        args.dilation_rate = 2
        self.conv_1 = self._build_conv(args)

        args.dilation_rate = 4
        self.conv_2 = self._build_conv(args)

        # post activation norm
        self.post_norm = RMSNorm(seq_len, eps)

        # feature collapse
        args.out_channels = 1
        args.kernel_size = 1
        args.dilation_rate = 1
        args.groups = 1
        self.logits = self._build_conv(args, False)

    def _build_conv(self, args: EGArgs, bias: bool = True) -> nn.Conv1d:
        return nn.Conv1d(
            args.in_channels,
            args.out_channels,
            args.kernel_size,
            padding=(args.kernel_size - 1) // 2 * args.dilation_rate,
            dilation=args.dilation_rate,
            groups=args.groups,
            bias=bias,
        )

    def _gate_out(self, x: Tensor):
        x = self.gate_branch(x.transpose(1, 2))
        return x.transpose(1, 2)

    def forward(self, number_log: Tensor, x: Tensor):
        # normalize input embeddings [pattern relationships]
        x_norm = self.pre_norm.forward(x, axis=1)

        # Dilated layer learning
        x_d = self.conv(x_norm) + self.conv_1(x_norm) + self.conv_2(x_norm)
        x_d = self.post_norm.forward(x_d * self._gate_out(x_norm), axis=1)

        # logits representation [including scale here - another run try this]
        # out = self.logits(number_log.unsqueeze(-1).unsqueeze(-1) * (x + x_d))
        out = self.logits((x + x_d))
        return out


class MetaDataTokens(nn.Module):
    """Takes a list of metadata and converts it to the required torch tensor"""

    def __init__(
        self,
        embedd_dim: int = 128,
        focus_data=None,
        eps: float = 1e-5,
        device=None,
    ) -> None:
        super().__init__()

        if focus_data is None:
            raise ValueError("Need to give a focus")

        focus_options = {"amount", "date", "number"}
        if focus_data.lower() not in focus_options:
            raise ValueError(f"Focus can only be {focus_options}")

        self.focus_data = focus_data.lower()

        self.__setup_holders(embedd_dim, focus_data)

        # Embedding layers for digits
        self.e_gen = EmbeddingGenerator(self.max_seq_len, eps)

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)

        self.device = device

    def __setup_holders(self, embed_dim, focus_data: str):
        key_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

        # Node date uses yyyy-mm-dd format
        key_list += [".", "-"] if focus_data in ["amount", "number"] else ["-"]

        self.num_embeddings = len(key_list)
        self.embedding_dim = embed_dim
        self.focus_data = focus_data

        # create key to index mapping
        self.idx = {k: v for v, k in enumerate(key_list)}

        if focus_data in ["amount", "number"]:
            self.max_seq_len = 20
        else:
            self.max_seq_len = 10

    def _token_size_manager(self, x: Tensor):
        if x.size(1) < self.max_seq_len:
            p_val = self.max_seq_len - x.size(1)
            x = torch.nn.functional.pad(x, (0, p_val, 0, 0), value=0)

        elif x.size(1) > self.max_seq_len:
            x = x[:, : self.max_seq_len]

        return x

    def forward(self, data: List[str]) -> Dict[str, Tensor]:
        # gather elements
        token_extractor: List[Tensor] = []
        values: List[float] = []

        # collecting all data information
        for s_val in data:
            # peroid marks scale for values
            if self.focus_data in ["amount", "number"]:
                values.append(abs(float(s_val)))
                if "." not in s_val:
                    s_val.append(".")

            else:
                parts = s_val.split("-")
                values.append(float(parts[0] + parts[1] + parts[2]))

            # limit to maximum length
            if len(s_val) > self.max_seq_len:
                s_val = s_val[: self.max_seq_len]

            token_indices = [self.idx[chr] for chr in s_val]
            tokens_tensor = torch.tensor(token_indices, dtype=torch.long)
            token_extractor.append(tokens_tensor)

        if token_extractor:
            val_tensor = torch.log(torch.tensor(values)).to(self.device)
            # Pad the sequences to have the same length
            t_tensors = torch.nn.utils.rnn.pad_sequence(
                token_extractor, batch_first=True, padding_value=0
            ).to(self.device)

            t_tensors = self._token_size_manager(t_tensors)

            # Embed the tokens
            embed_toks = self.embeddings(t_tensors)
            reps = self.e_gen(val_tensor, embed_toks)

            return {data[idx]: rep for idx, rep in enumerate(reps)}

        return {}
