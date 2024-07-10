import os

# Paths
RAW_DATA_PATH = os.path.join("data", "rastro.csv")
GEN_DATA_DIR = os.path.join("data", "samples")

# Data parameters
CROP_LENGTH = 50
CROP_STEP = 20
N_TO_PREDICT = 10

SPLITTING_SUBSET_SIZE = 1800

# 1 week of measurements
SAMPLES_PER_AGENT = 604800

# must sum to 1
SPLIT_SIZES = [0.8, 0.1, 0.1]

CONFIG = {
    "RAW_DATA_PATH": RAW_DATA_PATH,
    "GEN_DATA_DIR": GEN_DATA_DIR,
    "CROP_LENGTH": CROP_LENGTH,
    "CROP_STEP": CROP_STEP,
    "N_TO_PREDICT": N_TO_PREDICT,
    "SPLITTING_SUBSET_SIZE": SPLITTING_SUBSET_SIZE,
    "SAMPLES_PER_AGENT": SAMPLES_PER_AGENT,
    "SPLIT_SIZES": SPLIT_SIZES

}
