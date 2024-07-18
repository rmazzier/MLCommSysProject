from collections import OrderedDict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import Metrics
from flwr_datasets import FederatedDataset

from constants import DEVICE
from models import EncoderRNN, DecoderRNN
from train import train_epoch, test_step, setup_training

# disable_progress_bar()


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


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, config, trainloader, valloader, encoder, decoder, encoder_optimizer,):
        self.config = config
        self.trainloader = trainloader
        self.valloader = valloader
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = encoder_optimizer
        self.criterion = nn.L1Loss()

    def get_parameters(self):
        return get_parameters(self.net)

    def fit(self, parameters):
        set_parameters(self.net, parameters)
        train_epoch(
            self.trainloader,
            self.valloader,
            encoder=self.encoder,
            decoder=self.decoder,
            encoder_optimizer=self.encoder_optimizer,
            decoder_optimizer=self.decoder_optimizer,
            criterion=self.criterion,
            teacher_forcing=False,
            max_iters=100)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters):
        # Still wip

        # set_parameters(self.net, parameters)
        # loss, accuracy = test(self.net, self.valloader)
        # return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
        pass
