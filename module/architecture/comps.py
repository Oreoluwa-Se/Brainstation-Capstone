from ..preprocess.load_and_batch import MetaData
from .base import RMSNorm, ConvLayer, RepConvBlock, ConvNormAct
from dataclasses import dataclass
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
from typing import Optional, List, Dict
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torch


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
    use_amp: bool = False
    device: str = None


def make_divisible(n, div):
    if n % div == 0:
        return n

    return n + div - (n % div)


def precompute_rope_theta(
    head_dim: int, seq_len: int, device: str, theta: float = 10000.0
):
    assert head_dim % 2 == 0, "Head dimension must be divisible by 2"
    # Build the theta parameters
    # Shape: (head_dim / 2)
    theta_num = torch.arange(0, head_dim, 2).float()
    # Shape: (head_dim / 2)
    theta = 1.0 / (theta ** (theta_num / head_dim)).to(device)
    # Construct the positions (m parameter)
    # Shape: (seq_len)
    m = torch.arange(seq_len, device=device)
    # Outer product multplication to get all possible theta combination with each sequence position
    # Shape: (Seq_len, Head_dim/2)
    freqs = torch.outer(m, theta).float()
    # we use complex form for compact way of storing sin-cos
    # Shape: (Seq_len, Head_dim/2) -> (Seq_len, Head_dim/2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embeddings(x: Tensor, freqs_complex: Tensor, device: str):
    # (B, seq_len, h, head_dim) -> (B, seq_len, h, head_dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (seq_len, head_dim/2) -> (1, seq_len, 1, head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (B, seq_len, h, head_dim/2) * (1, seq_len, 1, head_dim/2) -> (B, seq_len, h, head_dim/2)
    x_rotated = x_complex * freqs_complex
    # (B, seq_len, h, head_dim/2) -> (B, seq_len, h, head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, seq_len, h, head_dim/2, 2) -> (B, seq_len, h, head_dim)
    x_out = x_out.reshape(*x.shape)

    return x_out.type_as(x).to(device)


class LocalQKV(nn.Module):
    """
    Peforms local aggregation using a convolutions then gets the q,k,v weights
    Then we use the linear layers to get the weights
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_pre_norm: bool = False,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        # Try to get better channel representation
        self.pc = RepConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=True,
            use_pre_norm=use_pre_norm,
            eps=eps,
        )

    def forward(self, x: Tensor) -> Tensor:
        x_c = self.pc(x)

        return x_c


class GlobalQKV(nn.Module):
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

    def forward(self, x: Tensor) -> Tensor:
        # shrink the channel space.
        x_norm = self.pre_norm(x)
        x_norm = self.gc(x_norm)

        return self.act(x_norm)


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    if n_rep == 1:
        return x

    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    # (B, Seq_len, n_kv_heads,1, head_dim)
    return (
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs, interim_dim: int = -1):
        super().__init__()

        # main dimensions
        self.chan_dim = interim_dim if interim_dim != -1 else args.dim
        self.head_dim = self.chan_dim // args.n_heads
        self.sqrt_head_dim = self.head_dim**0.5

        # potential setup for Grouped version
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        # ratio between number of heads for query and kv
        self.n_rep = self.n_heads_q // self.n_kv_heads

        # dimensions
        self.query_dim = int(0.5 * args.n_heads * self.head_dim)
        self.kv_dim = int(0.5 * self.n_kv_heads * self.head_dim)

        # qkv_local
        total_dim = self.query_dim + 2 * self.kv_dim
        self.lc_weights = nn.Linear(self.chan_dim, total_dim, bias=False)

        # weight mixer
        self.gc_weights = ConvLayer(
            self.chan_dim,
            total_dim,
            kernel_size=1,
            padding=0,
            bias=False,
        )

        # normalizations for query and key [pre attention]
        self.q_lc_norm = RMSNorm(self.query_dim, epsilon=args.norm_eps)
        self.k_lc_norm = RMSNorm(self.kv_dim, epsilon=args.norm_eps)

        self.q_gc_norm = RMSNorm(self.query_dim, epsilon=args.norm_eps)
        self.k_gc_norm = RMSNorm(self.kv_dim, epsilon=args.norm_eps)

        # attention dropout rate
        self.dropout = nn.Dropout(args.dropout_rate)

        # linear projections for local and global outputs
        self.local_out = nn.Linear(2 * self.query_dim, self.chan_dim)
        self.local_proj_dropout = nn.Dropout(args.proj_dropout_rate)
        self.lc_out_norm = RMSNorm(self.chan_dim, epsilon=args.norm_eps)

        self.global_out = nn.Linear(2 * self.query_dim, self.chan_dim)
        self.global_proj_dropout = nn.Dropout(args.proj_dropout_rate)
        self.gc_out_norm = RMSNorm(self.chan_dim, epsilon=args.norm_eps)

        self.amp = args.use_amp

    def _context_build(
        self, x: Tensor, weight_func: nn.Module, q_norm: RMSNorm, k_norm: RMSNorm
    ):
        batch_size, seq_len, _ = x.shape

        comb = weight_func(x)
        query, key, value = torch.split(
            comb, [self.query_dim, self.kv_dim, self.kv_dim], dim=-1
        )

        # (B, seq_len, 0.5 *h_q * Dim) -> (B, seq_len, h_q, 0.5 * Dim)
        query = q_norm(query).view(batch_size, seq_len, self.n_heads_q, -1)

        # (B, seq_len, 0.5 *h_kv * Dim) -> (B, seq_len, h_kv, 0.5 * Dim)
        key = k_norm(key).view(batch_size, seq_len, self.n_heads_q, -1)
        value = value.view(batch_size, seq_len, self.n_heads_q, -1)

        return query, key, value

    def _attention_forward(self, q, k, v, mask=None):
        # (B, head, seq_len, Dim) @ (B, head, Dim, seq_len) -> (B, head, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.sqrt_head_dim
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        return torch.matmul(attn, v)

    def _compute_attention(self, q_lc, k_lc, v_lc, q_gc, k_gc, v_gc, mask):
        # (B, seq_len, h_q, 0.5 * Dim) -> (B, seq_len, h_q, Dim)
        xq = self.q_rms_norm(torch.cat([q_lc, q_gc], dim=-1))
        # (B, seq_len, h_kv, 0.5 * Dim) -> (B, seq_len, h_kv, Dim)
        xk = self.k_rms_norm(torch.cat([k_lc, k_gc], dim=-1))
        xv = torch.cat([v_lc, v_gc], dim=-1)

        # if doing the grouped query version.. might need to repeat heads
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # ------> Now perform Multihead Attn <----------
        # (B, seq_len, head, Dim) -> (B, head, seq_len, Dim)
        xq, xk, xv = [tmp.transpose(1, 2) for tmp in (xq, xk, xv)]

        # (B, head, seq_len, head_dim)
        return checkpoint(self._attention_forward, xq, xk, xv, use_reentrant=False)

    def forward(
        self,
        local_c: Tensor,
        global_c: Tensor,
        freq_complex: Tensor = None,
        mask: Tensor = None,
    ):
        # This sequence calculates the
        batch_size, seq_len, _ = local_c.shape
        device = local_c.device
        use_auto_cast = self.amp and torch.cuda.is_available()

        with autocast(enabled=use_auto_cast):
            # ------> QKV Local <----------
            q_lc, k_lc, v_lc = self._context_build(
                local_c, self.lc_weights, self.q_lc_norm, self.k_lc_norm
            )

            # ------> QKV Global <----------
            q_gc, k_gc, v_gc = self._context_build(
                global_c, self.gc_weights, self.q_gc_norm, self.k_gc_norm
            )

            if freq_complex is not None:
                # including position encodings
                q_lc = apply_rotary_embeddings(q_lc, freq_complex, device)
                k_lc = apply_rotary_embeddings(k_lc, freq_complex, device)

                q_gc = apply_rotary_embeddings(q_gc, freq_complex, device)
                k_gc = apply_rotary_embeddings(k_gc, freq_complex, device)

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


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs, inter_channel: int = -1) -> None:
        super().__init__()

        dim = inter_channel if inter_channel != -1 else args.dim
        hidden_dim = 4 * dim
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
        swish = F.silu(self.w1(x), inplace=True)
        x_v = self.w3(x)
        x = swish * x_v
        x = self.w2(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, args=ModelArgs) -> None:
        super().__init__()

        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads

        self.pre_norm = RMSNorm(args.dim, epsilon=args.norm_eps)

        query_dim = args.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim

        inter_chan = (query_dim + 2 * kv_dim) // 3
        inter_chan = make_divisible(inter_chan, args.n_heads)

        # Context extractors
        self.lc = LocalQKV(args.dim, inter_chan, False, args.norm_eps)
        self.gc = GlobalQKV(args.dim, inter_chan, False, args.norm_eps)

        self.lc_norm = RMSNorm(inter_chan, epsilon=args.norm_eps)
        self.gc_norm = RMSNorm(inter_chan, epsilon=args.norm_eps)

        self.attn = SelfAttention(args, interim_dim=inter_chan)

        self.lc_ffn_norm = RMSNorm(inter_chan, epsilon=args.norm_eps)
        self.gc_ffn_norm = RMSNorm(inter_chan, epsilon=args.norm_eps)

        self.ff_lc = FeedForward(args)
        self.ff_gc = FeedForward(args)

        # final layer
        self.fin_layer = ConvNormAct(
            in_channels=2 * inter_chan,
            out_channels=args.dim,
            kernel_size=1,
            bias=True,
            eps=args.norm_eps,
        )

    def forward(self, x: Tensor, freq_complex: Tensor, mask: Tensor = None):
        # Building context windows
        x_norm = self.pre_norm(x)

        # (B, seq_len, Dim) -> (B, seq_len, inter_chan)
        local_c = self.lc(x_norm)
        lc_norm = self.lc_norm(local_c)

        global_c = self.gc(x_norm)
        gc_norm = self.gc_norm(global_c)

        # gets attention value for local context norm and global context norm
        lca_norm, gca_norm = self.attn.forward(lc_norm, gc_norm, freq_complex, mask)

        # residual connection with Mlp
        # (B, seq_len, inter_chan)
        local_c += self.ff_lc(self.lc_ffn_norm(lc_norm + lca_norm))
        global_c += self.ff_gc(self.gc_ffn_norm(gc_norm + gca_norm))

        # (B, seq_len, 2*inter_chan) -> # (B, seq_len, Dim)
        comb = self.fin_layer(torch.cat([local_c, global_c], dim=-1))

        # shortcut to previous layer
        return x + comb


class ConvAttention(nn.Module):
    def __init__(self, embed_dim, max_seq_len, device, eps):
        super(ConvAttention, self).__init__()
        self.conv1d = ConvLayer(
            in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, padding=1
        )

        self.pre_norm = RMSNorm(embed_dim, epsilon=eps)
        self.q_norm = RMSNorm(embed_dim, epsilon=eps)
        self.k_norm = RMSNorm(embed_dim, epsilon=eps)

        self.freqs_complex = precompute_rope_theta(
            embed_dim,
            max_seq_len,
            device=device,
        )

        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.embed_dim = embed_dim
        self.device = device

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        # x: embeddings from the embedding table
        # Assuming x is of shape [batch_size, seq_len, embed_dim]
        x_norm = self.pre_norm(self.conv1d(x))

        q = self.q_norm(self.query(x_norm)).view(batch_size, seq_len, 1, -1)  # Queries
        k = self.k_norm(self.key(x_norm)).view(batch_size, seq_len, 1, -1)  # Keys
        v = self.value(x_norm)  # Values

        # positional encoding
        q = apply_rotary_embeddings(q, self.freqs_complex[: x.shape[1]], self.device)
        k = apply_rotary_embeddings(k, self.freqs_complex[: x.shape[1]], self.device)

        q = q.view(batch_size, seq_len, -1)
        k = k.view(batch_size, seq_len, -1)

        # Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.embed_dim**0.5
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum to get the final output
        attended = torch.matmul(attn_weights, v)
        # adding the raw embeddings directly
        x += attended

        return x.mean(dim=1)


class Metadata2Token(nn.Module):
    """Takes a list of metadata and converts it to the required torch tensor"""

    def __init__(self, args: ModelArgs, focus_data=None) -> None:
        super().__init__()

        assert focus_data is not None, "Need to give a focus"
        assert focus_data.lower() in (
            "amount",
            "date",
            "number",
        ), "Focus can only be [amount, date, number]"

        # initial setup
        self.__setup_holders(args, focus_data)

        # Embedding layers for digits
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)

        # for when context is needed:
        self.c_embed = ConvAttention(
            self.embedding_dim, max_seq_len=30, device=args.device, eps=args.norm_eps
        )

    def __setup_holders(self, args: ModelArgs, focus_data: str):
        key_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        if focus_data.lower() == "amount":
            key_list += ["."]

        elif focus_data.lower() == "date":
            key_list += ["-"]  # yyyy-mm-dd format

        elif focus_data.lower() == "number":
            key_list += ["-"]

        self.num_embeddings = len(key_list)
        self.embedding_dim = args.dim
        self.focus_data = focus_data
        # create key to index mapping
        self.idx = {k: v for v, k in enumerate(key_list)}

    def forward(self, data: List[MetaData]):
        # gather elements
        token_extractor: List[Tensor] = []
        order_tracker: List[str] = []
        out_var: Dict[str, Tensor] = {}
        # max_len = 0
        for elem in data:
            c_list = elem[self.focus_data]
            for val in c_list:
                s_val = str(val)
                order_tracker.append(s_val)
                token_indices = [self.idx[char] for char in s_val]
                tokens_tensor = torch.tensor(token_indices, dtype=torch.long)
                token_extractor.append(tokens_tensor)

        if token_extractor:
            # Pad the sequences to have the same length
            token_tensor = torch.nn.utils.rnn.pad_sequence(
                token_extractor, batch_first=True, padding_value=0
            )
            # Embed the tokens
            embedded_tokens = self.embeddings(token_tensor)
            # Process through the ConvAttention
            reps = self.c_embed(embedded_tokens)

            out_var = {order_tracker[idx]: rep for idx, rep in enumerate(reps)}
        else:
            out_var = {}
        print(out_var)
        return out_var


# class Transformer(nn.Module):
#     """Transformer Block"""

#     def __init__(self, args: ModelArgs) -> None:
#         super().__init__()

#         assert args.vocab_size > 0, "Vocab size must be set"
#         self.args = args
#         self.vocab_size = args.vocab_size
#         self.n_layers = args.n_layers
#         self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

#         self.layers = nn.ModuleList()
#         for _ in range(args.n_layers):
#             self.layers.append(EncoderBlock(args))

#         self.norm = RMSNorm(args.dim, epsilon=args.norm_eps)

#         # final output layer
#         self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

#         # for computing frequency of rotary encodings
#         self.freqs_complex = precompute_rope_theta(
#             self.args.dim // self.args.n_heads,
#             self.args.seq_len,
#             device=self.args.device,
#         )

#     def forward(self, data: List[MetaData]):
#         # (B, seq_len)
#         pass


# class SelfAttention(nn.Module):
#     def __init__(self, args: ModelArgs) -> None:
#         super().__init__(*args, **kwargs)
