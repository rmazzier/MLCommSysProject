import datetime
import enum
import os
import matplotlib.pyplot as plt

import torch


class SPLIT(enum.Enum):
    TRAIN = "train"
    VALIDATION = "valid"
    TEST = "test"


def get_date_from_overalltime(overalltime):
    # Set the starting date as 29 june 2016 using the datetime module
    start_date = datetime.datetime(2016, 6, 29)

    # Get the first date which is the starting date plus the number of seconds
    date = start_date + datetime.timedelta(seconds=overalltime)
    return date


def plot_forecast_example(config, test_dataset, trained_encoder, trained_decoder, to_plot_idx=0):

    trained_encoder.eval()
    trained_decoder.eval()

    save_path = os.path.join(
        config["RESULTS_DIR"], config["MODEL_NAME"], f"forecast_example_{to_plot_idx}.png")
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        for sequence, target in test_dataset:
            input_tensor = sequence.unsqueeze(0).float().to(config.DEVICE)
            target_tensor = target.unsqueeze(0).float().to(config.DEVICE)

            encoder_outputs, encoder_hidden = trained_encoder(input_tensor)

            # No Teacher forcing
            decoder_outputs, _, _ = trained_decoder(
                encoder_outputs, encoder_hidden)

            # Define variables to plot
            last_elements = input_tensor[-config["N_TO_PREDICT"]:].cpu(
            ).numpy()
            ground_truth = target_tensor.cpu().numpy()
            forecast = decoder_outputs.cpu().numpy()

            xcoord_pred = list(range(len(last_elements) - 1,
                               len(last_elements) + len(ground_truth)))
            plt.plot(last_elements)
            plt.plot(xcoord_pred, [last_elements[-1]] +
                     ground_truth, 'd-', label='Ground truth')
            plt.plot(xcoord_pred, [last_elements[-1]] +
                     forecast, 'o--', label='Forecast',)
            plt.grid(alpha=0.25)

            plt.legend()

            plt.savefig(save_path)
            plt.show()
            break
