from src.client.client import Client
from src.client.cnn import Net

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms



import flwr as fl

DATA_PATH = "./data/cifar-10"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Load model and data
net = Net().to(DEVICE)
transform = transforms.Compose(
[transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
trainset = CIFAR10(".", train=True, download=True, transform=transform)
testset = CIFAR10(".", train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32)
num_examples = {"trainset" : len(trainset), "testset" : len(testset)}
client = Client(net, trainloader, testloader, num_examples)



fl.client.start_numpy_client(server_address="[::]:8080", client=client)