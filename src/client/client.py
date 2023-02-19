from collections import OrderedDict
from typing import Dict


from numpy import array
import torch
from torch.utils.data import DataLoader
from src.client.cnn import Net

import flwr as fl


class Client(fl.client.NumPyClient):
    __trainloader: DataLoader
    __testloader: DataLoader
    __net: Net
    __num_examples: Dict[str, int]
    __device: torch.device

    def __init__(self, trainloader: DataLoader, testloader: DataLoader, net: Net, num_examples: Dict[str, int]):
        """Load CIFAR-10 (training and test set)."""
        self.__trainloader = trainloader
        self.__testloader = testloader
        self.__net = net
        self.__num_examples = num_examples
    
   
    def __train(self, epochs: int):
        """Train the network on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.__net.parameters(), lr=0.001, momentum=0.9)
        for _ in range(epochs):
            for images, labels in self.__trainloader:
                images, labels = images.to(self.__device), labels.to(self.__device)
                optimizer.zero_grad()
                loss = criterion(self.__net(images), labels)
                loss.backward()
                optimizer.step()

    def __test(self):
        """Validate the network on the entire test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for data in self.__testloader:
                images, labels = data[0].to(self.__device), data[1].to(self.__device)
                outputs = self.__net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return loss, accuracy

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.__net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.__net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.__net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.__train(epochs=1)
        return self.get_parameters(config={}), self.__num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.__test()
        return float(loss), self.__num_examples["testset"], {"accuracy": float(accuracy)}
