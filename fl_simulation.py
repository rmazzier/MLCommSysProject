import flwr as fl
import wandb
import os
import json

from constants import DEVICE, CONFIG
from fl_setup import client_fn, FedAvgWandb, weighted_average_eval, weighted_average_fit


NUM_CLIENTS = 4
NUM_ROUNDS = 20

#

# Create FedAvg strategy
strategy = FedAvgWandb(
    my_config=CONFIG,
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
    min_fit_clients=4,  # Never sample less than 10 clients for training
    min_evaluate_clients=4,  # Never sample less than 5 clients for evaluation
    min_available_clients=4,  # Wait until all 10 clients are available
    fit_metrics_aggregation_fn=weighted_average_fit,
    evaluate_metrics_aggregation_fn=weighted_average_eval,

)

# Specify the resources each of your clients need. By default, each
# client will be allocated 1x CPU and 0x GPUs
client_resources = {"num_cpus": 1, "num_gpus": 0.0}
if DEVICE == "cuda":
    # here we are assigning an entire GPU for each client.
    client_resources = {"num_cpus": 1, "num_gpus": 1.0}
    # Refer to our documentation for more details about Flower Simulations
    # and how to setup these `client_resources`.

# Initialize wandb configuration dict

# PUT THIS RUN INSIDE THE "FedAvg" GROUP
CONFIG["WANDB_GROUP"] = "FedAvg"
CONFIG["MODEL_NAME"] = "FedAvg_" + CONFIG["MODEL_NAME"]

# For federated runs, the agent_idx does not make sense, so i remove it
if "AGENT_IDX" in CONFIG:
    del CONFIG["AGENT_IDX"]

# Save config.json file to folder
run_results_dir = os.path.join(CONFIG["RESULTS_DIR"], CONFIG["MODEL_NAME"])
os.makedirs(run_results_dir, exist_ok=True)
with open(os.path.join(run_results_dir, "config.json"), 'w') as f:
    json.dump(CONFIG, f)

# Start wandb session
wandb.login()
run = wandb.init(
    project="MLComSysProject",
    config=CONFIG,
    name=CONFIG["MODEL_NAME"],
    notes=CONFIG["NOTES"],
    reinit=True,
    mode=CONFIG["WANDB_MODE"],
    group=CONFIG["WANDB_GROUP"],
    tags=CONFIG["WANDB_TAGS"]
)

# Start the actual simulation
fl.simulation.start_simulation(
    client_fn=lambda x: client_fn(x, CONFIG),
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
    client_resources=client_resources,
)

run.finish()
