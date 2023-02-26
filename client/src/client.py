

from .model import CifarModel

import torch
import numpy as np
from torch.utils.data import DataLoader

from collections import OrderedDict
from typing import Dict, List, Tuple

from flwr import client
from flwr.common.logger import log
from logging import DEBUG, INFO



class CifarClient(client.NumPyClient):
    __trainloader: DataLoader
    __testloader: DataLoader
    __model: CifarModel
    __num_examples: Dict[str, int]
  

    def __init__(
        self,
        trainloader: DataLoader,
        testloader: DataLoader,
        num_examples: Dict,
        model: CifarModel
    ) -> None:
        self.__model = model
        self.__trainloader = trainloader
        self.__testloader = testloader
        self.__num_examples = num_examples

    

    def get_parameters(self, config) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.__model.net.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.__model.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.__model.net.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:

        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        log(INFO, f'Starting server {config["server_name"]} training round number {config["current_round"]} of {config["server_num_rounds"]}')
    
        self.__model.train(self.__trainloader, epochs=config["local_epochs"])
        return self.get_parameters(config), self.__num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        #log(INFO, f'Starting server {config["server_name"]} evaluation round number {config["current_round"]} of {config["server_num_rounds"]}')
        loss, accuracy = self.__model.test(self.__testloader)
        return float(loss), self.__num_examples["testset"], {"accuracy": float(accuracy)}