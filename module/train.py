from .architecture.base import RMSNorm
from .architecture.comps import make_divisible
from .architecture.dataclasses import ModelArgs
from .architecture.phase import FullProcess
from .estop import EarlyStopping
from .trainer import ModelTrainer
from module.preprocess.bpe import Encoder
from module.preprocess.load_and_batch import DataBatcher
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from typing import List
from utils import config
import math
import torch
import torch.nn as nn
import torch.optim as optim

torch.backends.cudnn.benchmark = True
torch.manual_seed(123)


def get_param_names(model: nn.Module, ignored_layers: List[nn.Module] = None):
    """
    Gathers the full names of parameters from  model, excluding ignored ones.

    Args:
    - model (nn.Module): The model from which to gather parameter names.
    - ignore_modules (list): module types to ignore.

    Returns:
    - List[str]: A list of parameter names.
    """
    if ignored_layers is None:
        ignored_layers = []

    ignore_types = tuple(ignored_layers)
    param_names = []

    for name, module in model.named_modules():
        if isinstance(module, ignore_types):
            continue

        for param_name, _ in module.named_parameters(recurse=False):
            full_name = f"{name}.{param_name}" if name else param_name
            param_names.append(full_name)

    return param_names


def create_optimizer(model: nn.Module = None):
    if model is None:
        return

    p_list = get_param_names(model, [nn.Embedding, RMSNorm])
    # remove biases
    p_list = [name for name in p_list if "bias" not in name]
    # group parameters
    optim_grouped_params = [
        {
            "params": [p for n, p in model.named_parameters() if n in p_list],
            "weight_decay": config.TRAINER["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in p_list],
            "weight_decay": 0.0,
        },
    ]

    return optim.Adam(optim_grouped_params, lr=config.TRAINER["lr"])


def initialize_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            # Calculate fan-in for uniform bias initialization
            if isinstance(module, (nn.Conv1d)):
                fan_in = module.in_channels * module.kernel_size[0]
            else:
                fan_in = module.in_features

            # Calculate bound for uniform initialization of bias
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(module.bias, -bound, bound)

    elif isinstance(module, nn.Embedding):
        dim = module.weight.data.size(1)
        nn.init.uniform_(module.weight, -1.0 / dim**0.5, 1.0 / dim**0.5)

    elif isinstance(module, RMSNorm):
        torch.nn.init.ones_(module.weight)


def verify_args(args: ModelArgs):
    """
    Verifies and adjusts the model arguments to ensure they meet the requirements for
    rotary position embeddings (ROPE)

    Parameters:
    args (ModelArgs): The model arguments object containing parameters like number of heads,
                      dimension sizes, and multipliers.
    """

    # Ensure n_kv_heads is even and n_heads is divisible by n_kv_heads if specified
    if args.n_kv_heads:
        args.n_kv_heads = make_divisible(args.n_kv_heads, 2)
        args.n_heads = make_divisible(args.n_heads, args.n_kv_heads)

    # Adjust the feedforward network dimension multiplier to be even
    if args.ffn_dim_multiplier:
        args.ffn_dim_multiplier = make_divisible(args.ffn_dim_multiplier, 2)

    # Adjust the model dimension to be divisible by 2 * n_heads for ROPE compatibility
    args.dim = make_divisible(args.dim, 2 * args.n_heads)

    print(f"Adjusted Parameters: \n{args}")


def dataset_loader() -> DataBatcher:
    # loading and batch preparation
    loader = DataBatcher()
    loader.load_state(config.BASE_LOCATION)

    # if nothing on file currently load
    if loader.nones:
        loader.load_and_process(
            base_loc=config.DATA_LOCATION,
            minority_loc=config.SPLIT_DATASETS + "target.csv",
            majority_loc=config.SPLIT_DATASETS
            + f"dataset_{config.DATASET_CONFIG['split_to_load']}.csv",
            train_test_split=config.DATASET_CONFIG["train_test_split"],
            validation_split=config.DATASET_CONFIG["validation_split"],
            training=config.DATASET_CONFIG["training_stage"],
        )

        loader.save_state(config.BASE_LOCATION)

    return loader


def basic_info(model: nn.Module):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    print(f"Total trainable parameters: {trainable_params}")
    print(f"Total non-trainable parameters: {non_trainable_params}")

    first_param_device = next(model.parameters()).device
    print(f"Model is on {first_param_device}")


def run():
    # num_iterations used for debugging
    model_storage = config.BASE_LOCATION + "model_info/"
    checkpoint_path = model_storage + "checkpth.pt"

    device = torch.device("cpu")
    if config.MODEL_ARGS["use_gpu"] and torch.cuda.is_available():
        device = torch.device("cuda:0")

    # Tokenizer
    tokenizer = Encoder(None)
    tokenizer.load_state(config.BASE_LOCATION)

    # model arguments
    args = ModelArgs(
        dim=config.MODEL_ARGS["embedding_dim"],
        n_layers=config.MODEL_ARGS["n_hidden_layers"],
        n_heads=config.MODEL_ARGS["n_heads"],
        n_kv_heads=config.MODEL_ARGS["n_kv_heads"],
        vocab_size=tokenizer.vocab_size,
        multiple_of=config.MODEL_ARGS["multiple_of"],
        ffn_dim_multiplier=config.MODEL_ARGS["ffn_multiplier"],
        norm_eps=config.MODEL_ARGS["norm_eps"],
        attn_dropout_rate=config.MODEL_ARGS["attn_dropout_rate"],
        proj_dropout_rate=config.MODEL_ARGS["proj_dropout_rate"],
        batch_size=config.DATASET_CONFIG["batch"],
        seq_len=config.MODEL_ARGS["max_seq_len"],
        num_target_classes=config.MODEL_ARGS["target_classes"],
        use_amp=config.MODEL_ARGS["use_amp"],
        device=device,
    )

    verify_args(args=args)

    # How the dataset is loaded
    loader = dataset_loader()
    # Tensor board runner
    writer = SummaryWriter("runs/baseline_run")

    # Model and weight initialization
    model = FullProcess(args, tokenizer)
    model.apply(initialize_weights)

    optimizer = create_optimizer(model)

    estop = EarlyStopping(
        min_delta=config.TRAINER["e_stop_min_delta"],
        patience=config.TRAINER["stopping_patience"],
        mode="min",
        verbose=False,
        checkpoint_path=checkpoint_path,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=2,
        cooldown=0,
        min_lr=1e-6,
        verbose=True,
    )

    # Loading trainer
    runner = ModelTrainer(
        model,
        optimizer,
        scheduler,
        loader=loader,
        estop=estop,
        writer=writer,
        device=device,
    )

    basic_info(model)
    runner.train(num_epochs=config.TRAINER["num_epochs"], verbose=True)


if __name__ == "__main__":
    run(1)  # for debugging
