from collections import OrderedDict
from typing import List, Tuple
import os

# import matplotlib.pyplot as plt
import flwr as fl
from flwr.common import Metrics
import numpy as np
import torch
import torch.nn as nn
import wandb

from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate
from flwr.server.client_manager import ClientManager
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns
from typing import List, Optional, Tuple, Callable, Dict, Union
from flwr.common import (
    FitRes,
    FitIns,
    MetricsAggregationFn,
    EvaluateRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)


from constants import DEVICE, CONFIG
from train import train_epoch, test_step, setup_training

"""
We need two helper functions to update the local model with parameters received from
the server and to get the updated model parameters from the local model:
set_parameters and get_parameters.
The following two functions do that.
"""


def set_parameters(net, parameters: List[np.ndarray]):
    """
    Set the parameters of the network.
    This is used to set the global parameters to a client.
    """
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    """
    Get the parameters of the network.
    Used when communicating from the client to the server.
    """

    return [val.cpu().numpy() for _, val in net.state_dict().items()]


"""
In Flower, clients are subclasses of flwr.client.Client or flwr.client.NumPyClient.
(I still don't know the difference between the two)

Then, we must implement the following three methods:
- get_parameters: Get the current model parameters.
- fit: Update the model using the provided parameters and return the updated parameters to the server
- evaluate: Evaluate the model using the provided parameters and return evaluation to the server.
"""


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, my_config, net, trainloader, valloader, optimizer, criterion, max_iters=100):
        self.my_config = my_config
        self.trainloader = trainloader
        self.valloader = valloader
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.max_iters = max_iters

    def get_parameters(self, config):
        print("Getting parameters!")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        print("Fitting client")
        set_parameters(self.net, parameters)
        train_loss, _, _ = train_epoch(
            self.my_config,
            self.trainloader,
            self.valloader,
            net=self.net,
            optimizer=self.optimizer,
            criterion=self.criterion,
            max_iters=100)
        return self.get_parameters({}), len(self.trainloader), {"train_loss": float(train_loss)}

    def evaluate(self, parameters, config):

        print("Evaluating client")

        set_parameters(self.net, parameters)
        valid_loss, valid_rmse = test_step(
            self.valloader, self.net, self.criterion)
        return float(valid_loss), len(self.valloader), {"valid_loss": float(valid_loss), "valid_rmse": float(valid_rmse)}


def client_fn(cid, config) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    print(f"Creating client {cid}")

    cid = int(cid)
    train_loader, valid_loader, _, _, net = setup_training(
        config, agent_idx=cid)

    # Load model
    net = net.to(DEVICE)

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=.001)

    # Loss function
    if config["CRITERION"] == "MSE":
        criterion = nn.MSELoss()
    elif config["CRITERION"] == "MAE":
        criterion = nn.L1Loss()
    else:
        raise ValueError("Invalid criterion")

    # Create a  single Flower client representing a single organization
    return FlowerClient(config,
                        net,
                        train_loader,
                        valid_loader,
                        optimizer,
                        criterion,
                        max_iters=20).to_client()


def weighted_average_eval(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    losses = [num_examples * m["valid_loss"] for num_examples, m in metrics]
    rmses = [num_examples * m["valid_rmse"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"valid_loss": sum(losses) / sum(examples),
            "valid_rmse": sum(rmses) / sum(examples)}


def weighted_average_fit(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"train_loss": sum(losses) / sum(examples)}


"""
Now I subclass flwr.server.strategy.FedAvg to be able to log things to wandb"""


class FedAvgWandb(fl.server.strategy.FedAvg):
    def __init__(self,
                 *,
                 my_config,
                 fraction_fit: float = 1.0,
                 fraction_evaluate: float = 1.0,
                 min_fit_clients: int = 2,
                 min_evaluate_clients: int = 2,
                 min_available_clients: int = 2,
                 evaluate_fn: Optional[
                     Callable[
                         [int, NDArrays, Dict[str, Scalar]],
                         Optional[Tuple[float, Dict[str, Scalar]]],
                     ]
                 ] = None,
                 on_fit_config_fn: Optional[Callable[[
                     int], Dict[str, Scalar]]] = None,
                 on_evaluate_config_fn: Optional[Callable[[
                     int], Dict[str, Scalar]]] = None,
                 initial_parameters: Optional[Parameters] = None,
                 accept_failures: bool = True,
                 fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
                 evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None
                 ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

        self.my_config = my_config

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, metrics_aggregated = super(
        ).aggregate_fit(server_round, results, failures)

        wandb.log(metrics_aggregated, step=server_round)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters)

            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")
            save_dir = os.path.join(
                self.my_config["RESULTS_DIR"], self.my_config["MODEL_NAME"])
            os.makedirs(save_dir, exist_ok=True)
            np.savez(os.path.join(
                save_dir, f"round-{server_round}-weights.npz"), *aggregated_ndarrays)

        return aggregated_parameters, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):
        print("Aggregating metrics")
        loss_aggregated, metrics_aggregated = super(
        ).aggregate_evaluate(server_round, results, failures)
        wandb.log(metrics_aggregated, step=server_round)
        print(loss_aggregated, metrics_aggregated)
        return loss_aggregated, metrics_aggregated


class FedProxWandb(fl.server.strategy.FedProx):
    def __init__(self,
                 *,
                 my_config,
                 fraction_fit: float = 1.0,
                 fraction_evaluate: float = 1.0,
                 min_fit_clients: int = 2,
                 min_evaluate_clients: int = 2,
                 min_available_clients: int = 2,
                 evaluate_fn: Optional[
                     Callable[
                         [int, NDArrays, Dict[str, Scalar]],
                         Optional[Tuple[float, Dict[str, Scalar]]],
                     ]
                 ] = None,
                 on_fit_config_fn: Optional[Callable[[
                     int], Dict[str, Scalar]]] = None,
                 on_evaluate_config_fn: Optional[Callable[[
                     int], Dict[str, Scalar]]] = None,
                 initial_parameters: Optional[Parameters] = None,
                 accept_failures: bool = True,
                 fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
                 evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
                 proximal_mu: float = 1.0,
                 ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            proximal_mu=proximal_mu
        )

        self.my_config = my_config

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, metrics_aggregated = super(
        ).aggregate_fit(server_round, results, failures)

        wandb.log(metrics_aggregated, step=server_round)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters)

            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")
            save_dir = os.path.join(
                self.my_config["RESULTS_DIR"], self.my_config["MODEL_NAME"])
            os.makedirs(save_dir, exist_ok=True)
            np.savez(os.path.join(
                save_dir, f"round-{server_round}-weights.npz"), *aggregated_ndarrays)

        return aggregated_parameters, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):
        print("Aggregating metrics")
        loss_aggregated, metrics_aggregated = super(
        ).aggregate_evaluate(server_round, results, failures)
        wandb.log(metrics_aggregated, step=server_round)
        print(loss_aggregated, metrics_aggregated)
        return loss_aggregated, metrics_aggregated
