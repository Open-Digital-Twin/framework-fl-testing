from .client import CifarClient
from .model import *
from .storage import StorageManager

import torch


from pathlib import Path
from os import environ, path

import torchvision
from flwr import client
from flwr.common.logger import log
from logging import INFO, ERROR
from time import sleep


CLIENT_ID = int(environ.get("CLIENT_ID"))
EXPERIMENT_NAME = environ.get("EXPERIMENT_NAME")
EXPERIMENT_PATH = environ.get("EXPERIMENT_PATH") 
MODEL = environ.get("MODEL")
DATASET_NAME = environ.get("DATASET_NAME")
environ['EXPERIMENT_RESULTS_PATH']= path.join(EXPERIMENT_PATH, EXPERIMENT_NAME,".temp","results",f"fl-framework-client-{str(CLIENT_ID)}" )


if environ.get('FL_CLIENT_DEVICE'):
    DEVICE: str = environ.get('FL_CLIENT_DEVICE')
else:
    DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


SERVER_ADDRESS = environ.get("SERVER_ADDRESS")
CERTIFICATES_PATH = environ.get("CERTIFICATES_PATH")



def main() -> None:
    loaded_storage = False
    storage = StorageManager(EXPERIMENT_PATH, EXPERIMENT_NAME, f"fl-framework-client-{str(CLIENT_ID)}",CLIENT_ID)
    if DATASET_NAME == "fmnist" :
        num_classes=10
        dim=1024
        dnn=1*28*28
        mid_dnn=100
        in_features=1
    elif DATASET_NAME == "cifar-10":
        num_classes=10
        dim=1600
        dnn=3*32*32
        mid_dnn=100
        in_features=3
    elif DATASET_NAME == "cifar-100":
        num_classes=100
        dim=1600
        dnn=3*32*32
        mid_dnn=100
        in_features=3
    else:
        raise NotImplementedError(f"Dataset {DATASET_NAME} is not implemented")
    
    if MODEL == "cnn":
        model = FedAvgCNN(in_features=in_features, num_classes=num_classes, dim=dim).to(DEVICE)
    elif MODEL == "dnn": # non-convex
        model = DNN(dnn, mid_dnn, num_classes=num_classes).to(DEVICE)
    elif MODEL == "resnet":
        model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes).to(DEVICE)

   
    elif MODEL == "googlenet":
        model = torchvision.models.googlenet(pretrained=False, aux_logits=False,num_classes=num_classes).to(DEVICE)

    else:
        raise NotImplementedError(f"Model {MODEL} is not implemented")
          
    sleep(60)
    while not loaded_storage:
        try:
            storage.initialize_experiment_path()
            trainloader, testloader, num_examples = storage.load_data(batch_size=10)
        except FileNotFoundError:
            sleep(60)
            continue
        loaded_storage = True

    RETRIES=5
    log(INFO, f"Loading data from: {EXPERIMENT_PATH}, Batch Number={CLIENT_ID}")
    
    
    log(INFO, f"Starting FL client")
    
    while(RETRIES > 0):
        try:
            client.start_client(
                server_address=SERVER_ADDRESS,
                root_certificates=Path(f"{CERTIFICATES_PATH}/ca.crt").read_bytes(),
                client=CifarClient(trainloader, testloader, num_examples, model=model, device=DEVICE).to_client()
                )
            break
        except Exception as e:
            RETRIES = RETRIES - 1
            log(ERROR, f"{e}")
            log(INFO, f"Retrying in 5 seconds")
            sleep(5)
    

if __name__ == "__main__":
    main()
