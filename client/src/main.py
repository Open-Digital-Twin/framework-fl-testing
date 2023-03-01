from .client import CifarClient
from .utils import isEmpty
from .model import CifarModel
from .dataset import CIFAR10_BATCH

from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms


from pathlib import Path
from os import environ

from flwr import client
from flwr.common.logger import log
from logging import INFO, ERROR
from time import sleep




if environ.get('FL_CLIENT_DEVICE'):
    DEVICE: str = environ.get('FL_CLIENT_DEVICE')
else:
    DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_PATH = environ.get("DATA_PATH")
SERVER_ADDRESS = environ.get("SERVER_ADDRESS")
CERTIFICATES_PATH = environ.get("CERTIFICATES_PATH")
DATA_BATCH_NUMBER = int(environ.get("DATA_BATCH_NUMBER"))

  
def load_data(DATA_PATH, DATA_BATCH_NUMBER):
    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
    trainset = CIFAR10_BATCH(DATA_PATH, train=True, download=isEmpty(DATA_PATH), transform=transform, batch=DATA_BATCH_NUMBER)
    testset = CIFAR10_BATCH(DATA_PATH, train=False, download=isEmpty(DATA_PATH), transform=transform, batch=DATA_BATCH_NUMBER)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)
    num_examples = {"trainset" : len(trainset), "testset" : len(testset)}
    return trainloader, testloader, num_examples

  

def main() -> None:
    RETRIES=5
    log(INFO,"Federated PyTorch Cifar-10 training")
    log(INFO, f"Loading data from: {DATA_PATH}, Batch Number={DATA_BATCH_NUMBER}")
    
    trainloader, testloader, num_examples = load_data(DATA_PATH, DATA_BATCH_NUMBER)
    log(INFO, f"Starting FL client")

    while(RETRIES > 0):
        try:
            client.start_numpy_client(
                server_address=SERVER_ADDRESS,
                root_certificates=Path(f"{CERTIFICATES_PATH}/ca.crt").read_bytes(),
                client=CifarClient(trainloader, testloader, num_examples, CifarModel(DEVICE))
                )
        except Exception as e:
            RETRIES = RETRIES - 1
            log(ERROR, f"{e}")
            log(INFO, f"Retrying in 5 seconds")
            sleep(5)
    

if __name__ == "__main__":
    main()
