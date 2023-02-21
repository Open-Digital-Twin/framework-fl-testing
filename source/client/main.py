from .clients import CifarClient

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from pathlib import Path

import os
from os import environ


import flwr as fl

DATA_PATH = "./data/cifar-10"
SERVER_ADDRESS = "127.0.0.1:4466"
#DEVICE: str = "cpu"
CERTIFICATES_PATH = "./.cache/certificates"

  
# Function to Check if the path specified
# specified is a valid directory
def isEmpty(path):
    if os.path.exists(path) and not os.path.isfile(path):
  
        # Checking if the directory is empty or not
        if not os.listdir(path):
            return True
        else:
            return False
    else:
        return True
  
  

def main() -> None:
    """Load data, start CifarClient."""

    # Load model and data
    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
    trainset = CIFAR10(DATA_PATH, train=True, download=isEmpty(DATA_PATH), transform=transform)
    testset = CIFAR10(DATA_PATH, train=False, download=isEmpty(DATA_PATH), transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)
    num_examples = {"trainset" : len(trainset), "testset" : len(testset)}
    client = CifarClient(trainloader, testloader, num_examples)



    fl.client.start_numpy_client(
        server_address=SERVER_ADDRESS,
        root_certificates=Path(f"{CERTIFICATES_PATH}/ca.crt").read_bytes(),
        client=client
        )

if __name__ == "__main__":
    main()
