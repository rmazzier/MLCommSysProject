
from dataset import Rastro_Dataset
from train_clients import train_all_clients
from fl_simulation import train_federated
from fl_setup import FedAvgWandb, weighted_average_fit, weighted_average_eval
import os

if __name__ == "__main__":

    from constants import CONFIG

    # Crop lenghts to test
    # crop_lengths = [10, 20, 30, 40, 50, 60]
    crop_lengths = [20, 30, 40, 50, 60]
    for crop_length in crop_lengths:
        print(f"Testing crop length {crop_length}")
        run_config = CONFIG.copy()
        run_config["CROP_LENGTH"] = crop_length
        run_config["MODEL_NAME"] = f"CL{crop_length}_{run_config['MODEL_NAME']}"

        # Generate the data
        Rastro_Dataset.generate_data(
            run_config, split_seed=123, standardize=True)

        strategy = FedAvgWandb(
            my_config=run_config,
            fraction_fit=1.0,  # Sample 100% of available clients for training
            fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
            min_fit_clients=4,  # Never sample less than 10 clients for training
            min_evaluate_clients=4,  # Never sample less than 5 clients for evaluation
            min_available_clients=4,  # Wait until all 10 clients are available
            fit_metrics_aggregation_fn=weighted_average_fit,
            evaluate_metrics_aggregation_fn=weighted_average_eval,
        )

        train_federated(run_config, strategy, n_rounds=100)
        # train_all_clients(run_config, Rastro_Dataset.generate_data)
