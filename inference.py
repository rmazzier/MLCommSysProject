import torch
import os
import json

from utils import plot_forecast_example, SPLIT
from train import setup_training
from dataset import Rastro_Dataset


def load_model(config, encoder, decoder):
    encoder_path = os.path.join(
        config["RESULTS_DIR"], config["MODEL_NAME"], "encoder.pt")

    decoder_path = os.path.join(
        config["RESULTS_DIR"], config["MODEL_NAME"], "decoder.pt")

    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))

    return encoder, decoder


if __name__ == "__main__":
    from constants import CONFIG, DEVICE

    model_name = "Seq2Seq_GRU_Att_02"

    # Load CONFIG json file
    with open(os.path.join(CONFIG["RESULTS_DIR"], model_name, "config.json"), 'r') as f:
        model_config = json.load(f)

    # Generate Data according to model_config
    Rastro_Dataset.generate_data(
        config=model_config, split_seed=123, standardize=True)

    _, _, test_dataset, test_loader, encoder, decoder = setup_training(
        model_config)

    # Load the model
    encoder, decoder = load_model(model_config, encoder, decoder)

    # Plot an example of forecasting from the test set
    plot_forecast_example(model_config, test_dataset,
                          encoder, decoder, to_plot_idx=0)
