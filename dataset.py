import os
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from inputimeout import inputimeout, TimeoutOccurred
from sklearn.model_selection import train_test_split
from utils import get_date_from_overalltime, SPLIT


def crop_with_step(sequence, crop_len, step):
    """Given a numpy array of shape (n, ...), it crops it into overlapping subportions of shape (crop_len, ...), with a sliding window of step.
    Final number of obtained arrays will be ceil((n-crop_len)/step).

    Parameters:
    :param sequence: The array to crop;
    :param crop_len: The width of the sliding window;
    :param step: The step of the sliding window"""
    idxs = np.arange(len(sequence) - crop_len, step=step)
    return np.array([sequence[idx: idx + crop_len] for idx in idxs])


class Rastro_Dataset(torch.utils.data.Dataset):
    def __init__(self, config, agent_idx, split):
        """
        Initialize the dataset with the given configuration and agent index.
        Parameters:
        :param config: The configuration dictionary;
        :param agent_idx: The index of the agent. If equal to -1 it will return the whole dataset;
        :param split: The split of the dataset (train, validation, test).   
        """
        self.config = config
        self.agent_idx = agent_idx
        self.split = split

        dirs = os.listdir(self.config["GEN_DATA_DIR"])
        # remove elements that are not directories
        dirs = [d for d in dirs if os.path.isdir(
            os.path.join(self.config["GEN_DATA_DIR"], d))]
        n_agents = len(dirs)

        # Check correctness of the agent index
        if self.agent_idx >= 0 and self.agent_idx <= n_agents:
            # Check whether the corresponding directory exists
            self.dataset_directory = os.path.join(
                self.config["GEN_DATA_DIR"], f"agent_{self.agent_idx}", self.split.value)

            # Check if there is at least one file in one of the subdirectories
            if len(os.listdir(os.path.join(self.dataset_directory))) == 0:
                raise FileNotFoundError(
                    f"It appears you have not generated the dataset. Please run Rastro_Dataset.generate_data(CONFIG)")

            self.filepaths = [os.path.join(
                self.dataset_directory, fname) for fname in os.listdir(self.dataset_directory)]

        elif self.agent_idx == -1:
            # Return the whole dataset
            self.dataset_directory = self.config["GEN_DATA_DIR"]
            self.filepaths = []
            for agent_idx in range(n_agents):
                agent_directory = os.path.join(
                    self.config["GEN_DATA_DIR"], f"agent_{agent_idx}", self.split.value)
                self.filepaths += [os.path.join(
                    agent_directory, fname) for fname in os.listdir(agent_directory)]
        else:
            raise ValueError(
                f"Agent index must be between 0 and {n_agents} or -1 to return the whole dataset")

    @staticmethod
    def clear_generated_files(config):
        """
        Clear all the generated files in the GEN_DATA_DIR.
        """

        # Check if the GEN_DATA_DIR exists
        if not os.path.exists(config["GEN_DATA_DIR"]):
            print("The directory does not exist. Exiting...")
            return

        # Clear all the files in the GEN_DATA_DIR

        # Loop over agent directories:
        for agent_dir in os.listdir(config["GEN_DATA_DIR"]):
            if agent_dir.endswith(".json"):
                os.remove(os.path.join(config["GEN_DATA_DIR"], agent_dir))
                continue
            train_directory = os.path.join(
                config["GEN_DATA_DIR"], agent_dir, "train")
            valid_directory = os.path.join(
                config["GEN_DATA_DIR"], agent_dir, "valid")
            test_directory = os.path.join(
                config["GEN_DATA_DIR"], agent_dir, "test")

            [[os.remove(os.path.join(d, f))
              for f in os.listdir(d)] for d in [train_directory, valid_directory, test_directory]]

        pass

    @staticmethod
    def interpolate_missing_values(data):

        # Check for missing values
        print("Percentage of missing values in the dataset: ",
              np.isnan(data).sum() / data.size)

        # Let's deal with those.
        # We will replace the missing values using a linear interpolation strategy.
        # def nan_helper(y):
        #     return np.isnan(y), lambda z: z.nonzero()[0]

        missing_value_dict = {}
        missing_value_dict[0] = 0

        for i in tqdm(range(1, data.shape[1])):
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

    @staticmethod
    def generate_data_simple(config, split_seed=123):
        """
        Also implement another data generation method, that doesnt use any overlapping in the
        cropping process. This is to avoid completely the problem of overlapping crops in different
        splits.
        In this way we have a tradeoff between the amount of data we can use for training and the bias
        injected in the data by performing the splitting-by-subset procedure.
        """

        # TODO finire di implementare

        # Now let's save the crops in different folders, one for each agent
        # Each agent must contain a total of n_agent_samples = 604800 (1 week of measurements)
        data_path = config["RAW_DATA_PATH"]
        data = pd.read_csv(data_path).to_numpy()

        # Percentage of entries with missing values
        data = Rastro_Dataset.interpolate_missing_values(data)

        # TODO: Add the time of day and day of the week here

        # standardize data
        data_std = (data - data.mean(axis=0)) / data.std(axis=0)
        # keep the first column (timestamps) unchanged
        data_std[:, 0] = data[:, 0]

        agent_idx = 0
        start_t = data[0, 0]
        agent_samples = []

        agent_directory = os.path.join(
            config["GEN_DATA_DIR"], f"agent_{agent_idx}")
        os.makedirs(agent_directory, exist_ok=True)

        for sample in enumerate(data_std):
            cur_t = sample[0]
            agent_crops.append(sample)

            if cur_t - start_t > config["SAMPLES_PER_AGENT"]:
                agent_crops = np.array(agent_crops)

                # perform splitting in train, validation and test using sklearn library
                train, tmp = train_test_split(
                    agent_crops, test_size=config["SPLIT_SIZES"][1] + config["SPLIT_SIZES"][2], random_state=split_seed)

                valid, test = train_test_split(
                    tmp, test_size=config["SPLIT_SIZES"][2], random_state=split_seed)

                # Save the crops in the corresponding directories
                np.save(os.path.join(agent_directory, "train_split.npy"), train)
                np.save(os.path.join(agent_directory, "valid.npy"), valid)
                np.save(os.path.join(agent_directory, "test.npy"), test)

                # Setup for next agent
                agent_idx += 1
                start_t = cur_t
                agent_crops = []

                agent_directory = os.path.join(
                    config["GEN_DATA_DIR"], f"agent_{agent_idx}")
                os.makedirs(agent_directory, exist_ok=True)
        pass

    @staticmethod
    def generate_data(config, split_seed=123):
        # Check if there is a config.json file in the GEN_DATA_DIR
        if os.path.exists(os.path.join(config["GEN_DATA_DIR"], "config.json")):
            with open(os.path.join(config["GEN_DATA_DIR"], "config.json"), "r") as f:
                current_config = json.load(f)

            # Check if the current config is the same as the one passed as argument
            if current_config == config:
                try:
                    choice = inputimeout(
                        prompt="There is already a dataset generated with the same configuration. Abort? (y/n) (10 seconds to answer)\n", timeout=10)
                except TimeoutOccurred:
                    choice = "y"

                if choice == "y":
                    print("Aborting...")
                    return

        # Clear all the files in the GEN_DATA_DIR
        Rastro_Dataset.clear_generated_files(config)

        # Start the data generation process by reading the .csv
        data_path = config["RAW_DATA_PATH"]
        data = pd.read_csv(data_path).to_numpy()

        # Percentage of entries with missing values
        data = Rastro_Dataset.interpolate_missing_values(data)

        # 1. Add new column for day of the week in last position
        data = np.insert(data, data.shape[1], 0, axis=1)
        print("Adding day of the week feature")
        for i in tqdm(range(len(data))):
            data[i, -1] = get_date_from_overalltime(data[i, 0]).weekday()

        # 2. Add new column for hour of the day also in last position
        data = np.insert(data, data.shape[1], 0, axis=1)
        print("Adding hour of the day feature")
        for i in tqdm(range(len(data))):
            data[i, -1] = get_date_from_overalltime(data[i, 0]).hour

        # standardize data
        print("Standardizing data...")
        data_std = (data - data.mean(axis=0)) / data.std(axis=0)
        # keep the first column (timestamps) unchanged
        data_std[:, 0] = data[:, 0]

        # We now subdivide the data to different agents
        # Remember: each agent must contain a total of n_agent_samples = config["SAMPLES_PER_AGENT"]

        # "normalize" the first column so that indexes start from 0
        print("Subdividing data into agents...")
        dataset_ids = data_std[:, 0] - data_std[0, 0]
        agent_idxs = dataset_ids // config["SAMPLES_PER_AGENT"]

        agents_data = []
        for agent_idx in range(int(agent_idxs.max())):
            agents_data.append(data_std[agent_idxs == agent_idx])

        for agent_idx, agent_data in tqdm(enumerate(agents_data)):
            print(f"Now generating data for agent {agent_idx}...")
            # create empty directories for train, valid and test sets
            train_directory = os.path.join(
                config["GEN_DATA_DIR"], f"agent_{agent_idx}", "train")
            valid_directory = os.path.join(
                config["GEN_DATA_DIR"], f"agent_{agent_idx}", "valid")
            test_directory = os.path.join(
                config["GEN_DATA_DIR"], f"agent_{agent_idx}", "test")

            os.makedirs(train_directory, exist_ok=True)
            os.makedirs(valid_directory, exist_ok=True)
            os.makedirs(test_directory, exist_ok=True)

            # Also save the current config file in the GEN_DATA_DIR as json file
            with open(os.path.join(config["GEN_DATA_DIR"], "config.json"), "w") as f:
                json.dump(config, f, indent=4)

            timespan_subsets = []

            # Do a similar operation, but this time we want to subdivide data of this
            # agent into subsets that span one SPLITTING_SUBSET_SIZE

            # normalize the first column so that indexes start from 0
            agent_samples_ids = agent_data[:, 0] - agent_data[0, 0]
            timespan_subset_ids = agent_samples_ids // config["SPLITTING_SUBSET_SIZE"]

            for i in range(int(timespan_subset_ids.max())):
                timespan_subsets.append(agent_data[timespan_subset_ids == i])

            # Now that we have divided everything in small subsets of 1 SPLITTING_SUBSET_SIZE each,
            # we randomly split this collection of subsets into train, validation and test

            train, tmp = train_test_split(
                timespan_subsets, test_size=config["SPLIT_SIZES"][1] + config["SPLIT_SIZES"][2], random_state=split_seed)

            valid, test = train_test_split(
                tmp, test_size=config["SPLIT_SIZES"][2], random_state=split_seed)

            # Now we actually crop and save the data in corresponding folders

            for sub_idx, train_subset in enumerate(train):
                # now perform cropping with step
                training_crops = crop_with_step(
                    train_subset, config["CROP_LENGTH"] + config["N_TO_PREDICT"], config["CROP_STEP"])

                # clear all previously existing files

                for k, crop in enumerate(training_crops):
                    # Save the crops in the corresponding directories
                    np.save(os.path.join(
                        train_directory, f"train_crop_agent_{agent_idx}_{sub_idx}_{k}"), crop)

            for sub_idx, valid_subset in enumerate(valid):
                # now perform cropping with step
                validation_crops = crop_with_step(
                    valid_subset, config["CROP_LENGTH"] + config["N_TO_PREDICT"], config["CROP_STEP"])

                for k, crop in enumerate(validation_crops):
                    # Save the crops in the corresponding directories
                    np.save(os.path.join(
                        valid_directory, f"valid_crop_agent_{agent_idx}_{sub_idx}_{k}"), crop)

            for sub_idx, test_subset in enumerate(test):
                # now perform cropping with step
                test_crops = crop_with_step(
                    test_subset, config["CROP_LENGTH"] + config["N_TO_PREDICT"], config["CROP_STEP"])

                for k, crop in enumerate(test_crops):
                    # Save the crops in the corresponding directories
                    np.save(os.path.join(
                        test_directory, f"test_crop_agent_{agent_idx}_{sub_idx}_{k}"), crop)

        # Print some statistics
        print("Number of agents: ", len(agents_data))

        # For each agent print the number of samples in train, validation and test
        for agent_idx, agent_data in enumerate(agents_data):
            train_samples = len(os.listdir(os.path.join(
                config["GEN_DATA_DIR"], f"agent_{agent_idx}", "train")))
            valid_samples = len(os.listdir(os.path.join(
                config["GEN_DATA_DIR"], f"agent_{agent_idx}", "valid")))
            test_samples = len(os.listdir(os.path.join(
                config["GEN_DATA_DIR"], f"agent_{agent_idx}", "test")))

            print(
                f"Agent {agent_idx}: Train samples: {train_samples}, Validation samples: {valid_samples}, Test samples: {test_samples}")

        print("Data generation completed")
        return

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        # Get the corresponding file
        filepath = self.filepaths[index]
        data = np.load(os.path.join(filepath))

        # get the input and target
        input = data[:self.config["CROP_LENGTH"]]
        target = data[self.config["CROP_LENGTH"]:]

        # Remove the features in the FEATURES_TO_REMOVE list plus the first index
        # discard first column (timestamps), and the features that we want to remove
        features_to_remove = [0]+self.config["FEATURES_TO_REMOVE"]
        input = np.delete(input, features_to_remove, axis=1)

        # The target should only contain features at indices 1,5,13
        target = target[:, [1, 5, 13]]

        return torch.tensor(input), torch.tensor(target)


if __name__ == "__main__":
    # Example usage of the Rastro_Dataset class
    from constants import CONFIG

    # Rastro_Dataset.generate_data(CONFIG, split_seed=123)
    train_dataset = Rastro_Dataset(CONFIG, agent_idx=0, split=SPLIT.TRAIN)

    cnt_train_dataset = Rastro_Dataset(
        CONFIG, agent_idx=-1, split=SPLIT.TRAIN)

    print(len(train_dataset))
    print(len(cnt_train_dataset))

    train_dataloader = torch.utils.data.DataLoader(
        cnt_train_dataset, batch_size=32, shuffle=True, num_workers=os.cpu_count())

    # do one iteration
    for input, target in train_dataloader:
        print(input.shape)
        print(target.shape)
        break
