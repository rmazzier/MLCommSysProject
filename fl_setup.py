from collections import OrderedDict
from typing import List

# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import flwr as fl


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
        train_loss, valid_loss = train_epoch(
            self.my_config,
            self.trainloader,
            self.valloader,
            net=self.net,
            optimizer=self.optimizer,
            criterion=self.criterion,
            max_iters=100)
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters, config):

        print("Evaluating client")

        set_parameters(self.net, parameters)
        valid_loss = test_step(self.valloader, self.net, self.criterion)
        return float(valid_loss), len(self.valloader), {"valid_loss": float(valid_loss)}
        pass


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
                        max_iters=100).to_client()
