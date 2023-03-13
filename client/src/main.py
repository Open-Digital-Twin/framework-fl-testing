from .client import CifarClient
from .model import CifarModel
from .storage import StorageManager

import torch


from pathlib import Path
from os import environ, path


from flwr import client
from flwr.common.logger import log
from logging import INFO, ERROR
from time import sleep



CLIENT_ID = int(environ.get("CLIENT_ID"))
EXPERIMENT_NAME = environ.get("EXPERIMENT_NAME")
EXPERIMENT_PATH = environ.get("EXPERIMENT_PATH") 
EXPERIMENT_RESULTS_PATH = path.join(EXPERIMENT_PATH, EXPERIMENT_NAME,".temp","results")


if environ.get('FL_CLIENT_DEVICE'):
    DEVICE: str = environ.get('FL_CLIENT_DEVICE')
else:
    DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


SERVER_ADDRESS = environ.get("SERVER_ADDRESS")
CERTIFICATES_PATH = environ.get("CERTIFICATES_PATH")



def main() -> None:
    loaded_storage = False
    storage = StorageManager(EXPERIMENT_PATH, EXPERIMENT_NAME, f"fl-framework-client-{str(CLIENT_ID)}",CLIENT_ID)
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
            client.start_numpy_client(
                server_address=SERVER_ADDRESS,
                root_certificates=Path(f"{CERTIFICATES_PATH}/ca.crt").read_bytes(),
                client=CifarClient(trainloader, testloader, num_examples, CifarModel(DEVICE))
                )
            break
        except Exception as e:
            RETRIES = RETRIES - 1
            log(ERROR, f"{e}")
            log(INFO, f"Retrying in 5 seconds")
            sleep(5)
    

if __name__ == "__main__":
    main()
