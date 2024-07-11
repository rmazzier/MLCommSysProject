import os

import torch
from train import train, setup_training, plot_forecast_example
from models import EncoderRNN, DecoderRNN
from dataset import Rastro_Dataset

if __name__ == "__main__":
    from constants import CONFIG
    from utils import SPLIT
    import json

    crop_lengths_to_test = [30, 50, 70]
    n_to_predict = [1, 5, 10]

    for crop_length in crop_lengths_to_test:
        for n_predict in n_to_predict:
            CONFIG["CROP_LENGTH"] = crop_length
            CONFIG["N_TO_PREDICT"] = n_predict

            CONFIG["MODEL_NAME"] = f"S2S_CL{crop_length}_NTP{n_predict}"
            CONFIG["NOTES"] = f"Testing crop length {crop_length} and n to predict {n_predict}"

            Rastro_Dataset.generate_data(config=CONFIG, split_seed=123)
            train_loader, valid_loader, test_dataset, test_loader, encoder, decoder = setup_training(
                CONFIG)
            train(
                config=CONFIG,
                train_dataloader=train_loader,
                valid_dataloader=valid_loader,
                test_dataloader=test_loader,
                encoder=encoder,
                decoder=decoder,
                learning_rate=0.001,
            )

            # save logs and models
            # Save logs and weights the trained models
            run_results_dir = os.path.join(
                CONFIG["RESULTS_DIR"], CONFIG["MODEL_NAME"])
            os.makedirs(run_results_dir, exist_ok=True)
            torch.save(encoder, os.path.join(run_results_dir, "encoder.pt"))
            torch.save(decoder, os.path.join(run_results_dir, "decoder.pt"))

            # Also save a copy of the relative config file
            with open(os.path.join(run_results_dir, "config.json"), 'w') as f:
                json.dump(CONFIG, f)

            # Plot an example of forecasting from the test set
            plot_forecast_example(test_dataset, encoder, decoder)
