from .dataclasses import ModelArgs, ConvInputs
from .comps import MetaDataTokens, EncoderBlock
from .comps import precompute_rope_theta
from .base import RMSNorm, GatedDilatedModule
from module.preprocess.bpe import Encoder
from module.preprocess.load_and_batch import BatchMeta, MetaData

# from .conv_types import ConvNormAct, MultiKConvBlock, ConvInputs, GatedConvNormAct
from torch import nn, Tensor
from typing import Optional, List, Dict
from .loss import Loss
import torch


class InfoPackage:
    def __init__(self) -> None:
        # Embeddings for different representations stored in a dictionary
        self.reps: Dict[str, Dict[str, Tensor]] = {}

    def add_rep(self, data_type: str, tensor: Tensor) -> None:
        if data_type in self.reps:
            self.reps[data_type] = tensor
        else:
            raise ValueError(f"Invalid data type: {data_type}")

    def get_rep(self, data_type: str, key: str) -> Tensor:
        if data_type in self.reps:
            return self.reps[data_type][key]
        else:
            raise ValueError(f"Invalid data type : {data_type} or key {key}")


class InputPhase(nn.Module):
    """
    This is the input learning phase that converts the table information
    into a tensor for the tranformer block
    """

    def __init__(self, args: ModelArgs, bpe: Encoder) -> None:
        super().__init__()

        self.num_rep = MetaDataTokens(
            embedd_dim=args.dim,
            focus_data="number",
            eps=args.norm_eps,
            device=args.device,
        )

        self.amnt_rep = MetaDataTokens(
            embedd_dim=args.dim,
            focus_data="amount",
            eps=args.norm_eps,
            device=args.device,
        )

        self.date_rep = MetaDataTokens(
            embedd_dim=args.dim,
            focus_data="date",
            eps=args.norm_eps,
            device=args.device,
        )

        self.bpe = bpe
        bpe.update_context_length(args.seq_len)

        self.meta_markers = bpe.get_meta_tokens()
        self.tag_to_text = {}
        self._extract_tags()

        # embeddings for text stuff
        self.embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.embedding_dim = args.dim
        self.max_seq_len = args.seq_len
        self.device = args.device

    def _extract_tags(self):
        for key, value in self.meta_markers.items():
            ext = key[2:-2]
            if ext.lower() == "num":
                ext = "number"

            self.tag_to_text[value] = ext.lower()

    def _collect_tensors(self, data_list: BatchMeta) -> InfoPackage:
        info = InfoPackage()

        # gets representative tensors
        info.reps["number"] = self.num_rep(data_list.get_array("number"))
        info.reps["amount"] = self.amnt_rep(data_list.get_array("amount"))
        info.reps["date"] = self.date_rep(data_list.get_array("date"))

        return info

    def forward(self, data_list: BatchMeta):
        info: InfoPackage = self._collect_tensors(data_list)

        # get base embeddings
        tokes_indicies_batch = [self.bpe.encode_text(text) for text in data_list.texts]

        tensors = torch.tensor(
            tokes_indicies_batch,
            dtype=torch.long,
            device=self.device,
        )

        # Perform batch embedding lookup
        batch_embeddings = self.embeddings(tensors)

        # sequentially swap specific embeddings as needed
        for batch_dim, token_idxs in enumerate(tokes_indicies_batch):
            metadata: MetaData = data_list.tracked_meta[batch_dim]

            for idx, val in enumerate(token_idxs):
                # teminate when we have no further metadata information
                if metadata.empty:
                    break

                if val in self.meta_markers.values():
                    key = metadata.get_front(self.tag_to_text[val])
                    found = info.get_rep(self.tag_to_text[val], str(key))
                    batch_embeddings[batch_dim, idx, :] = found

        return batch_embeddings


class IntermediateBlocks(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size > 0, "Vocab size must be set"
        self.max_seq_len = args.seq_len
        self.args = args

        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.layers = nn.ModuleList()
        args.downsample = True

        # Encoder blocks -> Downsample Block
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args, dwnsample=True))
            args.seq_len = args.seq_len // 2

        # Sequence Downsampling block ->
        self.chan_red = GatedDilatedModule(self._get_conv_args(args), pre_norm=True)

        # final output layer
        target = 1 if args.num_target_classes <= 2 else args.num_target_classes
        self.norm = RMSNorm(args.seq_len, epsilon=args.norm_eps)
        self.output = nn.Linear(args.seq_len, target, bias=False)

        # for computing frequency of rotary encodings
        self.freqs_complex = precompute_rope_theta(
            self.args.dim // self.args.n_heads,
            self.max_seq_len,
            device=self.args.device,
        )

    def _get_conv_args(self, args: ModelArgs):

        return ConvInputs(
            in_channels=args.dim,
            out_channels=1,  # summarize channels to 1 dimension
            kernel_size=5,
            bias=True,
            groups=1,
            dilation_rate=1,
            eps=args.norm_eps,
            act_out=True,
            along_sequence=False,
        )

    def forward(self, data: Tensor):
        batch = data.shape[0]
        for idx, layer in enumerate(self.layers):
            # (B, seq_len, dim)
            seq_len = int(self.max_seq_len / 2**idx)
            freqs_complex = self.freqs_complex[:seq_len]
            data = layer(data, freqs_complex)

        # learn and shrink across sequence length
        data = self.chan_red.forward(data).view(batch, -1)

        # get output logits
        return self.output(self.norm(data)).float()


class FullProcess(nn.Module):
    def __init__(self, args: ModelArgs, bpe: Encoder) -> None:
        super().__init__()

        # Table information -> Input Tokens
        self.pre_process = InputPhase(args, bpe)

        self.intermediate = IntermediateBlocks(args)

        # Loss calculation
        self.loss = Loss(temporal_weight=1.0, reduction="mean")

        self.device = args.device

    def forward(self, x: BatchMeta):
        out = self.pre_process(x)

        # run stuff across
        out = self.intermediate(out)

        # compute loss internally
        if len(x.target) == 0:
            return out, None

        # extract target values
        t_values = [[d["WEEK_NUM"], d["target"]] for d in x.target]
        y = torch.Tensor(t_values).type_as(out)

        # loss computation based on FocalBCE
        loss: Dict[str, float] = self.loss(out, y)

        # return out logits and loss
        return out, loss
