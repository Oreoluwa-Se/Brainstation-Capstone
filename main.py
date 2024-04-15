import torch

from module.train import run

# from module.architecture.conv_types import trial, multi

if __name__ == "__main__":
    seed_number = 42
    torch.manual_seed(seed_number)

    run()
