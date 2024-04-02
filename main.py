from module.architecture.comps import FeedForward, ModelArgs
from module.architecture.comps import repeat_kv, Metadata2Token
from module.preprocess.load_and_batch import MetaData
import torch.nn as nn
import torch
import random
from datetime import datetime, timedelta


def ff_trial(x: torch.Tensor, args: ModelArgs):
    print("Testing the Feed Forward layer")
    mod = FeedForward(args)

    print(f"Tensor shape pre FF: {x.shape}")
    res = mod(x)
    print(f"Tensor shape post FF: {res.shape}")

    res = repeat_kv(res, 2)
    print(f"Tensor shape post repeat FF: {res.shape}")


def generate_random_meta():
    # Function to generate a random date
    def random_date():
        start_date = datetime(2000, 1, 1)
        end_date = datetime(2023, 1, 1)
        time_between_dates = end_date - start_date
        random_number_of_days = random.randrange(time_between_dates.days)
        random_date = start_date + timedelta(days=random_number_of_days)
        return random_date.strftime("%Y-%m-%d")

    metadata_list = []
    for _ in range(10):
        md = MetaData()
        # Populate with random numbers and dates
        for _ in range(random.randint(1, 5)):  # Assuming 1 to 5 items per list
            md.insert_num(random.randint(0, 1000))
            md.insert_amount(random.randint(0, 10000))
            md.insert_date(random_date())
        metadata_list.append(md)

    return metadata_list


def meta_data_test(args):
    m_list = generate_random_meta()
    mod = Metadata2Token(args=args, focus_data="amount")

    mod(m_list)


def main():
    torch.manual_seed(0)

    batch = 1
    channels = 6
    seq_len = 12

    x_val = torch.Tensor(batch, seq_len, channels)
    args = ModelArgs(
        dim=channels,
        n_layers=1,
        n_heads=8,
        vocab_size=12,
        multiple_of=256,
        norm_eps=1e-5,
        attn_dropout_rate=0.1,
        proj_dropout_rate=0.5,
        batch_size=batch,
        seq_len=seq_len,
        use_amp=False,
        device=None,
    )

    # Feed forward trial
    # ff_trial(x_val, args)

    # metadata test
    meta_data_test(args)


if __name__ == "__main__":
    main()
