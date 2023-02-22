from collections import OrderedDict
from typing import Dict, List, Tuple


import numpy as np
from torch.utils.data import DataLoader
import torch
from os import environ
from .models import cifar as model


import flwr as fl

if environ.get('FL_CLIENT_DEVICE'):
    DEVICE: str = environ.get('FL_CLIENT_DEVICE')
else:
    DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CifarClient(fl.client.NumPyClient):
    __trainloader: DataLoader
    __testloader: DataLoader
    __net: model.Net
    __num_examples: Dict[str, int]
  

    def __init__(
        self,
        trainloader: DataLoader,
        testloader: DataLoader,
        num_examples: Dict,
        net: model.Net = model.Net().to(DEVICE)
    ) -> None:
        self.__net = net
        self.__trainloader = trainloader
        self.__testloader = testloader
        self.__num_examples = num_examples

    def get_parameters(self, config) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.__net.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.__net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.__net.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        print(config["batch_size"])  # Prints `32`
        print(config["current_round"])  # Prints `1`/`2`/`...`
        print(config["local_epochs"])  # Prints `2`
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        model.train(self.__net, self.__trainloader, epochs=config["local_epochs"], device=DEVICE)
        return self.get_parameters(config), self.__num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = model.test(self.__net, self.__testloader, device=DEVICE)
        return float(loss), self.__num_examples["testset"], {"accuracy": float(accuracy)}