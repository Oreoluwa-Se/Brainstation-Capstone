import regex as re

# storage locations
BASE_LOCATION = "D:/BrainStation/Project/Credit Risk Model/"
MAIN_TABLE_STORAGE = BASE_LOCATION + "developed/"

# configuration for byte pair encoding
BPE_CONFIG = {
    "Pattern": re.compile(""),
    "IQr_Mult": 1.5,
    "IQR_Iter": 5,
    "Compression_Ratio": 0.5,
}
