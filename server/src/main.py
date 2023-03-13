from . import strategies as custom_strategies

from pathlib import Path
from os import environ

from flwr import server
from flwr.common.logger import log
from flwr.common.logger import configure as configure_logger
from flwr.common import ndarrays_to_parameters, Metrics
from flwr.server import strategy as flwr_strategies
from logging import INFO
from typing import List, Tuple
from .dataset import Cifar10Dataset, Cifar100Dataset, FMNISTDataset
from .storage import StorageManager
import numpy as np
from .model import Net





EXPERIMENT_NAME = environ.get("EXPERIMENT_NAME")
EXPERIMENT_PATH = environ.get("EXPERIMENT_PATH")

SERVER_STRATEGY = environ.get("SERVER_STRATEGY")
SERVER_NUM_ROUNDS = int(environ.get("SERVER_NUM_ROUNDS"))
CERTIFICATES_PATH = environ.get("CERTIFICATES_PATH")
SERVER_ADDRESS=environ.get("SERVER_ADDRESS")
FRACTION_FIT=float(environ.get("FRACTION_FIT"))
MIN_FIT_CLIENTS=int(environ.get("MIN_FIT_CLIENTS"))
MIN_AVAILABLE_CLIENTS=int(environ.get("MIN_AVAILABLE_CLIENTS"))
CLIENT_LOCAL_EPOCHS=int(environ.get("CLIENT_LOCAL_EPOCHS"))
SERVER_NAME=environ.get("SERVER_NAME")
DATASET_NAME=environ.get("DATASET_NAME")
DATASET_BALANCE=bool(int(environ.get("DATASET_BALANCE")))
DATASET_PARTITION=environ.get("DATASET_PARTITION")
DATASET_NIID=bool(int(environ.get("DATASET_NIID")))
DATASET_BATCHES=int(environ.get("DATASET_BATCHES"))



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
        "current_round": server_round,
        "label_names":
    }
    return config

def get_parameters() -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in Net().state_dict().items()]

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}



def main() -> None:

   

    storage = StorageManager(EXPERIMENT_PATH, EXPERIMENT_NAME, SERVER_NAME)
    # Pass parameters to the Strategy for server-side parameter initialization
    
   
    
    if DATASET_NAME == "cifar-10":
        dataset = Cifar10Dataset(DATASET_BATCHES,partition=DATASET_PARTITION,balance=DATASET_BALANCE,niid=DATASET_NIID)
    elif DATASET_NAME == "cifar-100":
        dataset = Cifar100Dataset(DATASET_BATCHES,partition=DATASET_PARTITION,balance=DATASET_BALANCE,niid=DATASET_NIID)
    elif DATASET_NAME == "fmnist":
        dataset = FMNISTDataset(DATASET_BATCHES,partition=DATASET_PARTITION,balance=DATASET_BALANCE,niid=DATASET_NIID)
    else:
        raise NotImplementedError(f"Dataset {DATASET_NAME} is not implemented")
  
    if SERVER_STRATEGY == "flwr-fedyogi":
        strategy = flwr_strategies.FedYogi(
        initial_parameters=ndarrays_to_parameters(get_parameters()),
        fraction_fit=FRACTION_FIT,  # Sample % of available clients for the next round (0.1 = 10%)
        min_fit_clients=MIN_FIT_CLIENTS,  # Minimum number of clients to be sampled for the next round
        min_available_clients=MIN_AVAILABLE_CLIENTS,  # Minimum number of clients that need to be connected to the server before a training round can start
        on_fit_config_fn=fit_config, # The fit_config function we defined earlier
        on_evaluate_config_fn=evaluate_config,
        evaluate_metrics_aggregation_fn=weighted_average
    )
            
    elif SERVER_STRATEGY == "flwr-fedavg":
        strategy = flwr_strategies.FedAvg(
        fraction_fit=FRACTION_FIT,  # Sample % of available clients for the next round (0.1 = 10%)
        min_fit_clients=MIN_FIT_CLIENTS,  # Minimum number of clients to be sampled for the next round
        min_available_clients=MIN_AVAILABLE_CLIENTS,  # Minimum number of clients that need to be connected to the server before a training round can start
        on_fit_config_fn=fit_config, # The fit_config function we defined earlier
        on_evaluate_config_fn=evaluate_config,
        evaluate_metrics_aggregation_fn=weighted_average
        )
    
    elif SERVER_STRATEGY == "flwr-fedavgm":
        strategy = flwr_strategies.FedAvgM(
        fraction_fit=FRACTION_FIT,  # Sample % of available clients for the next round (0.1 = 10%)
        min_fit_clients=MIN_FIT_CLIENTS,  # Minimum number of clients to be sampled for the next round
        min_available_clients=MIN_AVAILABLE_CLIENTS,  # Minimum number of clients that need to be connected to the server before a training round can start
        on_fit_config_fn=fit_config, # The fit_config function we defined earlier
        on_evaluate_config_fn=evaluate_config,
        evaluate_metrics_aggregation_fn=weighted_average
        )
    elif SERVER_STRATEGY == "custom-aggregate":
         strategy = custom_strategies.AggregateCustomMetricStrategy(
        fraction_fit=FRACTION_FIT,  # Sample % of available clients for the next round (0.1 = 10%)
        min_fit_clients=MIN_FIT_CLIENTS,  # Minimum number of clients to be sampled for the next round
        min_available_clients=MIN_AVAILABLE_CLIENTS,  # Minimum number of clients that need to be connected to the server before a training round can start
        on_fit_config_fn=fit_config, # The fit_config function we defined earlier
        on_evaluate_config_fn=evaluate_config,
        evaluate_metrics_aggregation_fn=weighted_average
    )

    else:
        raise NotImplementedError(f"Server strategy {SERVER_STRATEGY} is not implemented")


    
    

  
    storage.start_experiment(dataset)

    log(INFO, f"Experiment {EXPERIMENT_NAME} started")

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