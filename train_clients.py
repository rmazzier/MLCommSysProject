import os
import json

from dataset import Rastro_Dataset
from train import setup_training, train
from utils import plot_predictions_long


def train_all_clients(config, data_gen_method):
    config["WANDB_GROUP"] = "Centralized"

    data_gen_method(
        config=config, split_seed=123, standardize=True)

    for agent_idx in range(4):
        train_agent_by_index(config, agent_idx)


def train_agent_by_index(config, agent_idx):
    agent_config = config.copy()
    agent_config["MODEL_NAME"] = f"AGT{agent_idx}_{agent_config['MODEL_NAME']}"
    agent_config["AGENT_IDX"] = agent_idx

    run_results_dir = os.path.join(
        agent_config["RESULTS_DIR"], agent_config["MODEL_NAME"])

    # Check if it exists:
    if os.path.exists(run_results_dir):
        print("Results directory already exists. Exiting...")
        return

    os.makedirs(run_results_dir, exist_ok=True)
    # Also save a copy of the relative agent_config file
    with open(os.path.join(run_results_dir, "config.json"), 'w') as f:
        json.dump(agent_config, f)

    train_loader, valid_loader, test_dataset, test_loader, net = setup_training(
        agent_config, agent_idx=agent_config["AGENT_IDX"])

    net = train(config=agent_config,
                train_dataloader=train_loader,
                valid_dataloader=valid_loader,
                test_dataloader=test_loader,
                net=net,
                learning_rate=0.0001)

    START_IDX = 100
    N_TO_PLOT = 200

    plot_predictions_long(agent_config, START_IDX,
                          N_TO_PLOT, test_dataset, net)


if __name__ == "__main__":
    from constants import CONFIG

    train_all_clients(CONFIG, Rastro_Dataset.generate_data)
    # Plot an example of forecasting from the test set
    # plot_forecast_example(test_dataset, encoder, decoder)
