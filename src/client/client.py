import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn

from collections import OrderedDict
from typing import Dict, List, Tuple

from flwr import client
from flwr.common.logger import log
from logging import DEBUG, INFO
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, f1_score
import pandas as pd
import matplotlib.pyplot as plt
from os import environ, makedirs, path

class CifarClient(client.NumPyClient):
    __trainloader: DataLoader
    __testloader: DataLoader
    __model: nn.Module
    __num_examples: Dict[str, int]
  

    def __init__(
        self,
        trainloader: DataLoader,
        testloader: DataLoader,
        num_examples: Dict,
        model: nn.Module,
        device: torch.device
    ) -> None:
        self.__model = model
        self.__trainloader = trainloader
        self.__testloader = testloader
        self.__num_examples = num_examples
        self.__device = device
    

    def get_parameters(self, config) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.__model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.__model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.__model.load_state_dict(state_dict, strict=True)



    def train(
        self,
        epochs: int
    ) -> None:
        """Train the network."""
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.__model.parameters(), lr=0.001, momentum=0.9)

        log(DEBUG, f"Training {epochs} epoch(s) w/ {len(self.__trainloader)} batches each")

        # Train the network
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.__trainloader, 0):
                images, labels = data[0].to(self.__device), data[1].to(self.__device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.__model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:  # print every 100 mini-batches
                    log(DEBUG,"[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0


    def test(
        self
    ) -> Tuple[float, float]:
        """Validate the network on the entire test set."""
        criterion = nn.CrossEntropyLoss()
        correct = 0
        total = 0
        loss = 0.0
        predicteds = []
        trues = []
        with torch.no_grad():
            for data in self.__testloader:
                images, labels = data[0].to(self.__device), data[1].to(self.__device)
                outputs = self.__model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                predicteds.extend(predicted)
                trues.extend(labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        log(DEBUG,f"accuracy = {accuracy}, correct = {correct}, total = {total}")
        return predicteds, trues, loss, accuracy
    

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:

        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        log(INFO, f'Starting server {config["server_name"]} training round number {config["current_round"]} of {config["server_num_rounds"]}')
    
        self.train( epochs=config["local_epochs"])
        return self.get_parameters(config), self.__num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        log(INFO, f'Starting server {config["server_name"]} evaluation round number {config["current_round"]} of {config["server_num_rounds"]}')
        predicteds, trues, loss, accuracy = self.test()
        self.plot_classification_report(predicteds, trues, config)
        return float(loss), self.__num_examples["testset"], {"accuracy": float(accuracy)}
    
    def plot_classification_report(self, trues, predicteds, config: Dict[str, str] = None):
        
       
        trues_list = []
        predicteds_list = []
        for tensor in trues:
            trues_list.append(tensor.item())
        for tensor in predicteds:
            predicteds_list.append(tensor.item())
        import warnings
        warnings.filterwarnings('ignore')
        
        y_true = np.array(trues_list).astype(int)
        y_pred = np.array(predicteds_list).astype(int)
        labels = np.arange(np.max(trues_list)+1)
        figure = plt.figure(figsize=(15, 15))
        target_names = labels.tolist()
        for i in range(len(target_names)):
            target_names[i] = str(target_names[i])
      

        clf_report = classification_report(y_true,
                                        y_pred,
                                        labels=labels,
                                        target_names=target_names,
                                        output_dict=True,
                                        zero_division=0)
        sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
        macro = f1_score(y_true, y_pred,labels=target_names, pos_label=1, average='macro', zero_division=0)
        micro = f1_score(y_true, y_pred,labels=target_names, pos_label=1, average='micro', zero_division=0)
        weighted = f1_score(y_true, y_pred,labels=target_names, pos_label=1, average='weighted', zero_division=0)

        log(INFO, f"f1_score: weighted={weighted}, micro={micro}, macro={macro}")
        
        dir_path = path.join(
                             environ.get("EXPERIMENT_RESULTS_PATH")
        )
        makedirs(dir_path,exist_ok=True)
        dir_path = path.join(dir_path, 
                             str(config["current_round"]) + ".png")
        plt.savefig(dir_path)