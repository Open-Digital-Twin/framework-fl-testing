from .clients import CifarClient

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from pathlib import Path
from . import utils

import os



import flwr as fl


DATA_PATH = os.environ.get("DATA_PATH")
SERVER_ADDRESS = os.environ.get("SERVER_ADDRESS")
CERTIFICATES_PATH = os.environ.get("CERTIFICATES_PATH")

  

  

def main() -> None:
    """Load data, start CifarClient."""

    # Load model and data
    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
    trainset = CIFAR10(DATA_PATH, train=True, download=utils.isEmpty(DATA_PATH), transform=transform)
    testset = CIFAR10(DATA_PATH, train=False, download=utils.isEmpty(DATA_PATH), transform=transform)
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
