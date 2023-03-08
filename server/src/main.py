from . import strategies

from pathlib import Path
from os import environ

from flwr import server
from flwr.common.logger import log
from flwr.common.logger import configure as configure_logger
from logging import INFO

from .dataset import Cifar10Dataset
from .storage import StorageManager




DEFAULT_LOG_DATA_PATH = "./experiments/server.log"
DEFAULT_EXPERIMENT_PATH = "./experiments/"
EXPERIMENT_NAME = environ.get("EXPERIMENT_NAME")

configure_logger("file", filename=DEFAULT_LOG_DATA_PATH)

SERVER_NUM_ROUNDS = int(environ.get("SERVER_NUM_ROUNDS"))
CERTIFICATES_PATH = environ.get("CERTIFICATES_PATH")
SERVER_ADDRESS=environ.get("SERVER_ADDRESS")
FRACTION_FIT=float(environ.get("FRACTION_FIT"))
MIN_FIT_CLIENTS=int(environ.get("MIN_FIT_CLIENTS"))
MIN_AVAILABLE_CLIENTS=int(environ.get("MIN_AVAILABLE_CLIENTS"))
CLIENT_LOCAL_EPOCHS=int(environ.get("CLIENT_LOCAL_EPOCHS"))
SERVER_NAME=environ.get("SERVER_NAME")


def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "server_num_rounds": SERVER_NUM_ROUNDS,
        "server_name": SERVER_NAME,
        "current_round": server_round,
        "local_epochs": CLIENT_LOCAL_EPOCHS,
    }
    return config

def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round."""
    config = {
        "server_num_rounds": SERVER_NUM_ROUNDS,
        "server_name": SERVER_NAME,
        "current_round": server_round
    }
    return config



def main() -> None:


    # Pass parameters to the Strategy for server-side parameter initialization
    
    strategy = strategies.AggregateCustomMetricStrategy(
        fraction_fit=FRACTION_FIT,  # Sample % of available clients for the next round (0.1 = 10%)
        min_fit_clients=MIN_FIT_CLIENTS,  # Minimum number of clients to be sampled for the next round
        min_available_clients=MIN_AVAILABLE_CLIENTS,  # Minimum number of clients that need to be connected to the server before a training round can start
        on_fit_config_fn=fit_config, # The fit_config function we defined earlier
        on_evaluate_config_fn=evaluate_config
    )

    storage = StorageManager(DEFAULT_EXPERIMENT_PATH, EXPERIMENT_NAME)
    dataset = Cifar10Dataset(10,partition='pat',balance=True,niid=True)
    

  
    storage.start_experiment(dataset)



    # Start SERVER

    server.start_server(
        server_address=SERVER_ADDRESS, 
       config=server.ServerConfig(num_rounds=SERVER_NUM_ROUNDS), 
       certificates=(
           Path(f"{CERTIFICATES_PATH}/ca.crt").read_bytes(),
           Path(f"{CERTIFICATES_PATH}/server.pem").read_bytes(),
           Path(f"{CERTIFICATES_PATH}/server.key").read_bytes()
      ),
       strategy=strategy)
    storage.end_experiment()

if __name__ == "__main__":
    main()