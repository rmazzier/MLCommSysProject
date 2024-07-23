import os

from dataset import Rastro_Dataset
from train import setup_training, train
from utils import plot_predictions_long

if __name__ == "__main__":
    from constants import CONFIG
    import json

    CONFIG["WANDB_GROUP"] = "Centralized"

    Rastro_Dataset.generate_data(
        config=CONFIG, split_seed=123, standardize=True)

    for agent_idx in range(4):
        CONFIG["MODEL_NAME"] = f"AGT{agent_idx}_{CONFIG['MODEL_NAME']}"
        CONFIG["AGENT_IDX"] = agent_idx

        run_results_dir = os.path.join(
            CONFIG["RESULTS_DIR"], CONFIG["MODEL_NAME"])
        os.makedirs(run_results_dir, exist_ok=True)
        # Also save a copy of the relative config file
        with open(os.path.join(run_results_dir, "config.json"), 'w') as f:
            json.dump(CONFIG, f)

        train_loader, valid_loader, test_dataset, test_loader, net = setup_training(
            CONFIG, agent_idx=CONFIG["AGENT_IDX"])

        net = train(config=CONFIG,
                    train_dataloader=train_loader,
                    valid_dataloader=valid_loader,
                    test_dataloader=test_loader,
                    net=net,
                    learning_rate=0.0001)

        START_IDX = 100
        N_TO_PLOT = 200

        plot_predictions_long(CONFIG, START_IDX, N_TO_PLOT, test_dataset, net)
        # Plot an example of forecasting from the test set
        # plot_forecast_example(test_dataset, encoder, decoder)
