import flwr as fl
from typing import List,Tuple
from pathlib import Path

SERVER_NUM_ROUNDS = 10
CERTIFICATES_PATH = "./.cache/certificates"



def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "batch_size": 32,
        "current_round": server_round,
        "local_epochs": 3,
    }
    return config

class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ):
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)
        print(f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}")

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)
    
 

def main() -> None:
    # Pass parameters to the Strategy for server-side parameter initialization
    
    strategy = AggregateCustomMetricStrategy(
        fraction_fit=0.5,  # Sample % of available clients for the next round (0.1 = 10%)
        min_fit_clients=2,  # Minimum number of clients to be sampled for the next round
        min_available_clients=4,  # Minimum number of clients that need to be connected to the server before a training round can start
        on_fit_config_fn=fit_config # The fit_config function we defined earlier
    )

  



    # Start SERVER

    fl.server.start_server(
        server_address="127.0.0.1:4466", 
        config=fl.server.ServerConfig(num_rounds=SERVER_NUM_ROUNDS), 
        certificates=(
            Path(f"{CERTIFICATES_PATH}/ca.crt").read_bytes(),
            Path(f"{CERTIFICATES_PATH}/server.pem").read_bytes(),
            Path(f"{CERTIFICATES_PATH}/server.key").read_bytes()
        ),
        strategy=strategy)

if __name__ == "__main__":
    main()