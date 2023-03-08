
import ujson
import numpy as np
from sklearn.model_selection import train_test_split
from os import path, makedirs
from .dataset import Dataset
import numpy as np
from typing import Optional, Dict
from flwr.common.logger import log
from flwr.common.logger import configure as configure_logger
from logging import INFO

import shutil

    


class StorageManager():

    __experiment_path: str
    __experiment_name: str
    __use_cache: bool
    __experiment_config: dict
   


    def __init__(self, experiment_path:str, experiment_name: str, use_cache_always: bool = True):
        self.__experiment_path = experiment_path
        self.__experiment_name= experiment_name
        self.experiment_config= {"dataset": {},
                        "experiment": self.__experiment_name,
                        "current_run_number": 0
                        }
        self.__use_cache = use_cache_always
        

    def __check_experiment_path(self, file_name:  Optional[str] = None):
        if file_name:
            return path.exists(self.__experiment_path + file_name)
        else:
            return path.exists(self.__experiment_path)
        
   
    

    
    def initialize_experiment_path(self: str):
        log(INFO, f"Initializing experiment path: {self.__experiment_path + self.__experiment_name}")
        if not self.__check_experiment_path():
            makedirs(self.__experiment_path)
        if not self.__check_experiment_path(self.__experiment_name):
            makedirs(self.__experiment_path + f"{self.__experiment_name}/runs")
        if not self.__check_experiment_path(f"{self.__experiment_name}/runs"):
            makedirs(self.__experiment_path + f"{self.__experiment_name}/runs")
        if not self.__check_experiment_path(f"{self.__experiment_name}/logs"):
            makedirs(self.__experiment_path + f"{self.__experiment_name}/logs")
        if not self.__check_experiment_path(f"{self.__experiment_name}/data"):
            makedirs(self.__experiment_path + f"{self.__experiment_name}/data")
        shutil.rmtree(self.__experiment_path + f"{self.__experiment_name}/data/train", ignore_errors=True)
        makedirs(self.__experiment_path + f"{self.__experiment_name}/.temp")
        
        
            

    def load_experiment_config(self):
        
        if self.__check_experiment_path(f"{self.__experiment_name}/config.json"):
            with open(self.__experiment_path  + f"{self.__experiment_name}/config.json", 'r') as f:
                self.experiment_config = ujson.load(f)
                log(INFO, f"Found experiment config {self.__experiment_path + self.__experiment}")
            return True
        else:
            log(INFO, f"Missing experiment config file")
            return False
 
    


    
    
    def __check_current_dataset_config(self,  dataset: Dataset):
        # check existing dataset
            if self.experiment_config['dataset'] == dataset.get_config():
                return True
            else:
                return False


    def __separate_data(self, dataset: Dataset) -> bool:
        X = [[] for _ in range(dataset.num_clients)]
        y = [[] for _ in range(dataset.num_clients)]
        statistic = [[] for _ in range(dataset.num_clients)]

        dataset_content, dataset_label = dataset.get_rawdata()

        dataidx_map = {}

    

        if dataset.partition == 'pat':
            idxs = np.array(range(len(dataset_label)))
            idx_for_each_class = []
            for i in range(dataset.num_classes):
                idx_for_each_class.append(idxs[dataset_label == i])

            class_num_per_client = [dataset.class_per_client for _ in range(dataset.num_clients)]
            for i in range(dataset.num_classes):
                selected_clients = []
                for client in range(dataset.num_clients):
                    if class_num_per_client[client] > 0:
                        selected_clients.append(client)
                    selected_clients = selected_clients[:int(dataset.num_clients/dataset.num_classes*dataset.class_per_client)]

                num_all_samples = len(idx_for_each_class[i])
                num_selected_clients = len(selected_clients)
                num_per = num_all_samples / num_selected_clients
                if dataset.balance:
                    num_samples = [int(num_per) for _ in range(num_selected_clients-1)]
                else:
                    num_samples = np.random.randint(max(num_per/10, dataset.least_samples/dataset.num_classes), num_per, num_selected_clients-1).tolist()
                num_samples.append(num_all_samples-sum(num_samples))

                idx = 0
                for client, num_sample in zip(selected_clients, num_samples):
                    if client not in dataidx_map.keys():
                        dataidx_map[client] = idx_for_each_class[i][idx:idx+num_sample]
                    else:
                        dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx+num_sample], axis=0)
                    idx += num_sample
                    class_num_per_client[client] -= 1

        elif dataset.partition == "dir":
            # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
            min_size = 0
            K = dataset.num_classes
            N = len(dataset_label)

            while min_size < dataset.least_samples:
                idx_batch = [[] for _ in range(dataset.num_clients)]
                for k in range(K):
                    idx_k = np.where(dataset_label == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(dataset.alpha, dataset.num_clients))
                    proportions = np.array([p*(len(idx_j)<N/dataset.num_clients) for p,idx_j in zip(proportions,idx_batch)])
                    proportions = proportions/proportions.sum()
                    proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

            for j in range(dataset.num_clients):
                dataidx_map[j] = idx_batch[j]
        else:
            raise NotImplementedError

        # assign data
        for client in range(dataset.num_clients):
            idxs = dataidx_map[client]
            X[client] = dataset_content[idxs]
            y[client] = dataset_label[idxs]

            for i in np.unique(y[client]):
                statistic[client].append((int(i), int(sum(y[client]==i))))
                

        del dataset_content, dataset_label
        # gc.collect()

        for client in range(dataset.num_clients):
            print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
            print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
            print("-" * 50)

        return X, y, statistic

        


    def __split_data(self, X, y, dataset: Dataset) -> tuple:
        # Split dataset
        log(INFO, f"Distributing the dataset to {dataset.num_clients} clients.")
        train_data, test_data = [], []
        num_samples = {'train':[], 'test':[]}

        for i in range(len(y)):
            X_train, X_test, y_train, y_test = train_test_split(
                X[i], y[i], train_size=dataset.train_size, shuffle=True)

            train_data.append({'x': X_train, 'y': y_train})
            num_samples['train'].append(len(y_train))
            test_data.append({'x': X_test, 'y': y_test})
            num_samples['test'].append(len(y_test))

        print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
        print("The number of train samples:", num_samples['train'])
        print("The number of test samples:", num_samples['test'])
        print()
        del X, y
        # gc.collect()

        return train_data, test_data


    def update_experiment_config(self, key, value ):
        self.experiment_config.update({key: value})
    
    def save_experiment_config(self):
        log(INFO, f"Saving experiment config: {self.__experiment_path + self.__experiment_name}")
        with open(self.__experiment_path + f"{self.__experiment_name}" + "/config.json", 'w') as f:
            ujson.dump(self.experiment_config, f)
        

    def __save_data_distribution(self, train_data, test_data):       
        # gc.collect()
        log(INFO, f"Saving data distribution")
        for idx, train_dict in enumerate(train_data):
            with open(self.__experiment_path + f"{self.__experiment_name}/data/train/" + str(idx) + '.npz', 'wb') as f:
                np.savez_compressed(f, data=train_dict)
        for idx, test_dict in enumerate(test_data):
            with open(self.__experiment_path + f"{self.__experiment_name}/data/test/" + str(idx) + '.npz', 'wb') as f:
                np.savez_compressed(f, data=test_dict)
        log(INFO, f"Data distribution saved")

    def check_dataset(self, dataset: Dataset):

        log(INFO, f"Checking if the experiment dataset configuration changed.")
        if self.__check_current_dataset_config(dataset) and self.__use_cache:
            log(INFO, f"The experiment dataset configuration has not changed.")
            log(INFO, f"The experiment will run with last configuration")
            return True
        else:
            log(INFO, f"The experiment dataset configuration has changed.")
            log(INFO, f"The experiment will run with new configuration")
        return False
    
    def __delete_distribution(self):
        shutil.rmtree(self.__experiment_path + f"{self.__experiment_name}/data/train", ignore_errors=True)
        shutil.rmtree(self.__experiment_path + f"{self.__experiment_name}/data/test", ignore_errors=True)

    def __distribute_dataset(self, dataset: Dataset):
        
        X, y, statistic = self.__separate_data(dataset)
        train_data, test_data = self.__split_data(X, y, dataset)
        self.__delete_distribution()
        if not self.__check_experiment_path(f"{self.__experiment_name}/data/test"):
            makedirs(self.__experiment_path + f"{self.__experiment_name}/data/test")
        if not self.__check_experiment_path(f"{self.__experiment_name}/data/train"):
            makedirs(self.__experiment_path + f"{self.__experiment_name}/data/train")
        self.__save_data_distribution(train_data, test_data)
        self.update_experiment_config('statistic', statistic)
        self.update_experiment_config('dataset', dataset.get_config())

    def start_experiment(self, dataset):
        self.initialize_experiment_path()
        if self.load_experiment_config():
            if self.check_dataset(dataset):
                return
            else:
                self.__distribute_dataset(dataset)
        else:
            self.__distribute_dataset(dataset)
        self.update_experiment_config("current_run_number", self.__experiment_config['current_run_number'] + 1)

    def end_experiment(self):
        shutil.move(f"{self.__experiment_path}{self.__experiment_name}/.temp", f"{self.__experiment_path}{self.__experiment_name}/runs/{self.__experiment_config['current_run_number']}")
        self.save_experiment_config()

        
        
        
        

    
        
            
                       
