import datetime
import enum
import os
import matplotlib.pyplot as plt
from constants import DEVICE
import numpy as np

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


def plot_forecast_example(config, test_dataset, trained_net, to_plot_idx=0, suffix=""):

    trained_net.eval()

    save_path = os.path.join(
        config["RESULTS_DIR"], config["MODEL_NAME"], f"forecast_example_{to_plot_idx}_{suffix}.png")
    # os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        # Load the original sequence
        filepath = test_dataset.filepaths[to_plot_idx]
        data = np.load(filepath)

        # get the input and target
        input_raw = data[:config["CROP_LENGTH"]]
        target_raw = data[config["CROP_LENGTH"]:]

        # Remove the features in the FEATURES_TO_REMOVE list plus the first index
        # discard first column (timestamps), and the features that we want to remove
        features_to_remove = [0]+config["FEATURES_TO_REMOVE"]
        tmp = np.delete(input_raw, features_to_remove, axis=1)

        # The target should only contain features at indices 1,5,13
        target = target_raw[:, [1, 5, 13]]

        # To gpu
        input_tensor = torch.tensor(
            tmp).unsqueeze(0).float().to(DEVICE)
        target_tensor = torch.tensor(
            target).unsqueeze(0).float().to(DEVICE)

        predictions = trained_net(input_tensor)

        # Squeeze and convert to numpy (shape (10,3))
        predictions = predictions.squeeze().cpu().numpy()
        ground_truth = target_tensor.squeeze().cpu().numpy()
        sequence = input_raw[:, [1, 5, 13]]

        # Plot
        fig, ax = plt.subplots(3, 1, figsize=(10, 20))
        for i in range(3):
            ax[i].plot(sequence[:, i], label="Input", color="blue")
            ax[i].plot(
                range(config["CROP_LENGTH"], config["CROP_LENGTH"]+config["N_TO_PREDICT"]), ground_truth[:, i], label="Ground Truth", color="green")
            ax[i].plot(
                range(config["CROP_LENGTH"], config["CROP_LENGTH"]+config["N_TO_PREDICT"]), predictions[:, i], label="Prediction", color="red")
            ax[i].set_title(f"Feature {i}")
            ax[i].legend()

        plt.savefig(save_path)
        plt.close()
        # plt.show()


def plot_predictions_long():
    # TODO: implement this
    pass


def interpolate_missing_values(data):
    """
    Replaces all missing values in the dataset using a linear interpolation strategy."""

    # Check for missing values
    print("Percentage of missing values in the dataset: ",
          np.isnan(data).sum() / data.size)

    # Let's deal with those.
    # We will replace the missing values using a linear interpolation strategy.
    # def nan_helper(y):
    #     return np.isnan(y), lambda z: z.nonzero()[0]

    missing_value_dict = {}
    missing_value_dict[0] = 0

    for i in range(1, data.shape[1]):
        n_nan = np.isnan(data[:, i]).sum()
        missing_value_dict[i] = n_nan

        y = data[:, i]
        # nans, x = nan_helper(y)
        nans = np.isnan(y)
        def x(z): return z.nonzero()[0]
        y[nans] = np.interp(x(nans), x(~nans), y[~nans])
        data[:, i] = y

    # for the last row, replace the missing values with the previous value
    for i in range(1, data.shape[1]):
        if np.isnan(data[-1, i]):
            data[-1, i] = data[-2, i]

    # Check that no missing values are left
    assert np.isnan(data).sum() == 0
    print("All missing values have been replaced")

    for i in range(0, data.shape[1]):
        print("Feature", i, "had", missing_value_dict[i], "missing values")
    return data


def interp_discontinuities(data, max_interp_width):
    new_data = []
    new_data.append(data[0])

    for i in range(1, len(data)):
        if i % 100000 == 0:
            print(f"Checking row {i}")
        if data[i, 0] == data[i-1, 0] + 1:
            new_data.append(data[i])
        elif data[i, 0] - data[i-1, 0] <= max_interp_width:
            for j in range(int(data[i-1, 0] + 1), int(data[i, 0])):
                new_row = np.zeros(data.shape[1])
                new_row[0] = j
                new_row[1:] = [np.nan for _ in range(data.shape[1] - 1)]
                new_data.append(new_row)
            new_data.append(data[i])

    new_data = np.array(new_data)
    return new_data
