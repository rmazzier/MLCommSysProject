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
    "RESULTS_DIR": os.path.join("results"),

    # Dataset parameters
    "CROP_LENGTH": 60,
    "CROP_STEP": 10,
    "N_TO_PREDICT": 10,
    "SPLITTING_SUBSET_SIZE": 900,
    "SAMPLES_PER_AGENT": 604800,  # 1 week of measurements
    "SPLIT_SIZES": [0.8, 0.1, 0.1],  # must sum to 1
    "FEATURES_TO_REMOVE": [3, 4, 6, 7, 11, 12],
    # "FEATURES_TO_REMOVE": [],
    "MAX_INTERP_WIDTH": 3,
    # RNN parameters
    "CRITERION": "MSE",
    "EPOCHS": 100,
    "BATCH_SIZE": 128,
    "HIDDEN_SIZE": 64,
    "TEACHER_FORCING": True,
    "NUM_LAYERS": 3,
    "CELL_TYPE": "GRU",
    "BIDIRECTIONAL": False,
    "ATTENTION": True,

    # --- WANDB VARIABLES ---
    "MODEL_NAME": "S2S_GRU_Att_CL60CS10_TF_MSE",
    "WANDB_MODE": "online",
    # "WANDB_MODE": "disabled",
    "NOTES": "With MSE",

}
