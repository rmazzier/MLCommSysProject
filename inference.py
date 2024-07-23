import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from utils import plot_forecast_example, SPLIT, load_model, plot_predictions_long, load_model_from_npz
from train import setup_training
from dataset import Rastro_Dataset


if __name__ == "__main__":
    from constants import CONFIG, DEVICE

    # MODEL_NAME = "S2S_GRU_Att_CL60CS10_TF_MAE_V2"
    MODEL_NAME = "FedAvg_S2S_GRU_Att_CL60CS10_TF_MAE_V2"

    START_IDX = 100
    N_TO_PLOT = 100

    # Load CONFIG json file
    with open(os.path.join(CONFIG["RESULTS_DIR"], MODEL_NAME, "config.json"), 'r') as f:
        model_config = json.load(f)

    # Generate Data according to model_config
    # But before that, this time I want to force the Crop_Step to be 10
    # This is because I want to compare ALL the ground truth values with ALL the predictions
    model_config["CROP_STEP"] = 10
    Rastro_Dataset.generate_data(
        config=model_config, split_seed=123, standardize=True)

    # Setup training
    _, _, test_dataset, _, net = setup_training(model_config)

    # Load the model weights
    if "Fed" in MODEL_NAME:
        net = load_model_from_npz(model_config, net)
        pass
    else:
        net = load_model(model_config, net)

    for i in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:

        plot_predictions_long(model_config, i,
                              N_TO_PLOT, test_dataset, net)
