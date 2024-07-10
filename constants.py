import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
# RAW_DATA_PATH = os.path.join("data", "rastro.csv")
# GEN_DATA_DIR = os.path.join("data", "samples")

# # Data parameters
# CROP_LENGTH = 50
# CROP_STEP = 20
# N_TO_PREDICT = 10

# SPLITTING_SUBSET_SIZE = 1800

# # 1 week of measurements
# SAMPLES_PER_AGENT = 604800

# # must sum to 1
# SPLIT_SIZES = [0.8, 0.1, 0.1]

# EPOCHS = 10
# BATCH_SIZE = 32
# HIDDEN_SIZE = 256

CONFIG = {
    # Paths
    "RAW_DATA_PATH": os.path.join("data", "rastro.csv"),
    "GEN_DATA_DIR": os.path.join("data", "samples"),

    # Data parameters
    "CROP_LENGTH": 50,
    "CROP_STEP": 20,
    "N_TO_PREDICT": 10,
    "SPLITTING_SUBSET_SIZE": 1800,
    "SAMPLES_PER_AGENT": 604800,  # 1 week of measurements
    "SPLIT_SIZES": [0.8, 0.1, 0.1],  # must sum to 1
    "EPOCHS": 10,
    "BATCH_SIZE": 64,
    "HIDDEN_SIZE": 512
}
