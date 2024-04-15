import regex as re

# storage locations
DATA_LOCATION = "
BASE_LOCATION = "
SPLIT_DATASETS = BASE_LOCATION + "split_sets/"
MAIN_TABLE_STORAGE = BASE_LOCATION + "developed/"
MODEL_STORAGE = BASE_LOCATION + "model/"

# Configuration for BytePairEncoder
BPE_CONFIG = {
    "Pattern": None,
    "IQr_Mult": 1.5,
    "IQR_Iter": 5,
    "Compression_Ratio": 0.5,
}

# Configuration for Dataset Configuration
DATASET_CONFIG = {
    "split_to_load": 1,
    "train_test_split": 0.7,
    "validation_split": 0.7,
    "training_stage": True,
    "batch": 128,
}

# Configuration for Transformer
MODEL_ARGS = {
    "embedding_dim": 64,  # Channels for embedding dimensions
    "n_hidden_layers": 1,  # Number of intermediate blocks
    "n_heads": 4,  # Number of attention heads
    "n_kv_heads": None,  # Number of heads for key and value [if different should be a multiple of n_heads]
    "multiple_of": 128,  # Internal channels are multiples of this number
    "ffn_multiplier": None,  # expansion factor in feed-forward layer
    "norm_eps": 1e-5,
    "attn_dropout_rate": 0.1,
    "proj_dropout_rate": 0.5,
    "max_seq_len": 1024,
    "target_classes": 2,  # Output classes
    "use_amp": True,  # use automatic precision
    "use_gpu": True,  # currently implemented for single GPU
}

# Trainer Parameters
TRAINER = {
    "lr": 1e-4,  # learning rate
    "num_epochs": 12,
    "weight_decay": 0.0,
    "warmup_patience": 2,
    "reduction_patience": 5,
    # Early stopping parameters
    "e_stop_min_delta": 0.001,
    "stopping_patience": 10,
    "save_checkpoint": 5,
    "eval_every": 1,
}
