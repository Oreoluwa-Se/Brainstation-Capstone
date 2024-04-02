import regex as re

# storage locations
DATA_LOCATION = "D:/BrainStation/Project/Credit Risk Model/"
BASE_LOCATION = "D:/wslShared/Brainstation-Capstone/"
MAIN_TABLE_STORAGE = BASE_LOCATION + "developed/"

# configuration for byte pair encoding
BPE_CONFIG = {
    "Pattern": None,
    "IQr_Mult": 1.5,
    "IQR_Iter": 5,
    "Compression_Ratio": 0.5,
}
