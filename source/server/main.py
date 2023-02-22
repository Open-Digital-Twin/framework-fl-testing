import flwr as fl
from pathlib import Path
from os import environ

from . import strategies

SERVER_NUM_ROUNDS = int(environ.get("SERVER_NUM_ROUNDS"))
CERTIFICATES_PATH = environ.get("CERTIFICATES_PATH")
SERVER_ADDRESS=environ.get("SERVER_ADDRESS")
FRACTION_FIT=float(environ.get("FRACTION_FIT"))
MIN_FIT_CLIENTS=int(environ.get("MIN_FIT_CLIENTS"))
MIN_AVAIABLE_CLIENTS=int(environ.get("MIN_AVAIABLE_CLIENTS"))

def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "batch_size": 32,
        "current_round": server_round,
        "local_epochs": 3,
    }
    return config



def main() -> None:
    # Pass parameters to the Strategy for server-side parameter initialization
    
    strategy = strategies.AggregateCustomMetricStrategy(
        fraction_fit=FRACTION_FIT,  # Sample % of available clients for the next round (0.1 = 10%)
        min_fit_clients=MIN_FIT_CLIENTS,  # Minimum number of clients to be sampled for the next round
        min_available_clients=MIN_AVAIABLE_CLIENTS,  # Minimum number of clients that need to be connected to the server before a training round can start
        on_fit_config_fn=fit_config # The fit_config function we defined earlier
    )

  



    # Start SERVER

    fl.server.start_server(
        server_address=SERVER_ADDRESS, 
        config=fl.server.ServerConfig(num_rounds=SERVER_NUM_ROUNDS), 
        certificates=(
            Path(f"{CERTIFICATES_PATH}/ca.crt").read_bytes(),
            Path(f"{CERTIFICATES_PATH}/server.pem").read_bytes(),
            Path(f"{CERTIFICATES_PATH}/server.key").read_bytes()
        ),
        strategy=strategy)

if __name__ == "__main__":
    main()