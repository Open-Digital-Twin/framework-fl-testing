""" Import libraries """
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple
from abc import ABC, abstractmethod
from flwr.common.logger import log
from logging import DEBUG


class Net(nn.Module):
    """Define class for neural network initialization"""
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    """Define function to pass into neural network next layer"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x  

class CNNModel(ABC):
  
    @abstractmethod
    def train(
        self,
        trainloader: torch.utils.data.DataLoader,
        epochs: int,
    ) -> None:
        pass

    @abstractmethod
    def test(
        self,
        testloader: torch.utils.data.DataLoader,
    ) -> Tuple[float, float]:
        pass
                                            
class CifarModel(CNNModel):
    
    net: Net = Net()

    def __init__(self, device) -> None:
        self.net.to(device)
        self.device = device

    def train(
        self,
        trainloader: torch.utils.data.DataLoader,
        epochs: int,
    ) -> None:
        """Train the network."""
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

        log(DEBUG, f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")

        # Train the network
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                images, labels = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:  # print every 100 mini-batches
                    log(DEBUG,"[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0


    def test(
        self,
        testloader: torch.utils.data.DataLoader,
    ) -> Tuple[float, float]:
        """Validate the network on the entire test set."""
        criterion = nn.CrossEntropyLoss()
        correct = 0
        total = 0
        loss = 0.0
        predicteds = []
        trues = []
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                predicteds.extend(predicted)
                trues.extend(labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        log(DEBUG,f"accuracy = {accuracy}, correct = {correct}, total = {total}")
        return predicteds, trues, loss, accuracy
