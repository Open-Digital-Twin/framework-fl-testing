
import ujson
import numpy as np
from sklearn.model_selection import train_test_split
from os import path, makedirs
from .dataset import Dataset
import random
import numpy as np
from typing import Optional, Dict
from flwr.common.logger import log
from flwr.common.logger import configure as configure_logger
from logging import INFO

import shutil

random.seed(1)
np.random.seed(1)


class StorageManager():

    __experiment_path: str
    __experiment_base_path: str
    __experiment_name: str
    __use_cache: bool
    __experiment_config: dict
    __hostname: str
   


    def __init__(self, experiment_base_path:str, experiment_name: str, hostname: str,use_cache_always: bool = True):
        self.__experiment_base_path = path.join(experiment_base_path)
        self.__experiment_path = path.join(experiment_base_path, experiment_name)
        self.__experiment_name= path.join(experiment_name)
        self.__hostname = hostname
        self.__experiment_config= {"dataset": {},
                        "experiment": self.__experiment_name,
                        "current_run_number": 0
                        }
        self.__use_cache = use_cache_always
        

    def __check_path(self, path_str: str):
        return path.exists(path.join(path_str))
        
    

    
    def initialize_experiment_path(self):
        
        
        if not self.__check_path(self.__experiment_path):
            makedirs(self.__experiment_path)

        if not self.__check_path(path.join(self.__experiment_path, "runs")):
            makedirs(path.join(self.__experiment_path, "runs"))

        if not self.__check_path(path.join(self.__experiment_path, "data")):
            makedirs(path.join(self.__experiment_path, "data"))

        shutil.rmtree(path.join(self.__experiment_path, ".temp"), ignore_errors=True)
        makedirs(path.join(self.__experiment_path, ".temp"))
        makedirs(path.join(self.__experiment_path, ".temp","logs"))
        makedirs(path.join(self.__experiment_path, ".temp","results"))
        file = path.join(self.__experiment_path,".temp","logs",f"{self.__hostname}.log")
        print(file)
        with open(file, mode='w+') as files:
            pass
        configure_logger("file", filename=file)
       
        log(INFO, f"Initializing experiment path: {self.__experiment_path}")

        
        
        


    def load_experiment_config(self):
        experiment_config_path = path.join(self.__experiment_path, "config.json")
        if self.__check_path(experiment_config_path):
            with open(experiment_config_path, 'r') as f:
                self.__experiment_config = ujson.load(f)
                log(INFO, f"Found experiment config {experiment_config_path}")
            return True
        else:
            log(INFO, f"Missing experiment config file")
            return False
 
    


    
    
    def __check_current_dataset_config(self,  dataset: Dataset):
        # check existing dataset
            if self.__experiment_config['dataset'] == dataset.get_config():
                return True
            else:
                return False


    def __separate_data(self, dataset: Dataset) -> bool:
        # separate data using the dataset config
        X = [[] for _ in range(dataset.num_clients)]
        y = [[] for _ in range(dataset.num_clients)]
        statistic = [[] for _ in range(dataset.num_clients)]

        # read data from disk
        log(INFO, f"Reading data from disk")
        dataset_content, dataset_label = dataset.get_rawdata()

        dataidx_map = {}

    

        if dataset.partition == 'pat':
            #Divide data by classes using the dataset config
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

        log(INFO, f"Total number of samples: {sum(num_samples['train'] + num_samples['test'])}")
        log(INFO,f"The number of train samples: {num_samples['train']}")
        log(INFO,f"The number of test samples: {num_samples['test']}")

        del X, y
        # gc.collect()

        return train_data, test_data


    def update_experiment_config(self, key, value ):
        self.__experiment_config.update({key: value})
    
    def save_experiment_config(self):
        
        log(INFO, f"Saving experiment config {self.__experiment_path}")
        with open(path.join(self.__experiment_path, "config.json"), 'w') as f:
            ujson.dump(self.__experiment_config, f)
        

    def __save_data_distribution(self, train_data, test_data):       
        # gc.collect()
        log(INFO, f"Saving data distribution")
        for idx, train_dict in enumerate(train_data):
            with open(path.join(self.__experiment_path, "data","train", str(idx)+'.npz'), 'wb') as f:
                np.savez_compressed(f, data=train_dict)
        for idx, test_dict in enumerate(test_data):
            with open(path.join(self.__experiment_path, "data","test", str(idx)+'.npz'), 'wb') as f:
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
        shutil.rmtree(path.join(self.__experiment_path, "data","train"), ignore_errors=True)
        shutil.rmtree(path.join(self.__experiment_path, "data","test"), ignore_errors=True)

    def __distribute_dataset(self, dataset: Dataset):
        
        X, y, statistic = self.__separate_data(dataset)
        train_data, test_data = self.__split_data(X, y, dataset)
        self.__delete_distribution()
        if not self.__check_path(path.join(self.__experiment_path, "data","train")):
            makedirs(path.join(self.__experiment_path, "data","train"))
        if not self.__check_path(path.join(self.__experiment_path, "data","test")):
            makedirs(path.join(self.__experiment_path, "data","test"))
        self.__save_data_distribution(train_data, test_data)
        self.update_experiment_config('statistic', statistic)
        self.update_experiment_config('dataset', dataset.get_config())

    def start_experiment(self, dataset):
        self.initialize_experiment_path()
        if self.load_experiment_config():
            if self.check_dataset(dataset):
                pass
            else:
                self.__distribute_dataset(dataset)
        else:
            self.__distribute_dataset(dataset)
        self.update_experiment_config("current_run_number", self.__experiment_config['current_run_number'] + 1)
        

    def end_experiment(self):
        log(INFO, f"Saving experiment {self.__experiment_path}")
        shutil.move(path.join(self.__experiment_path, ".temp"),path.join(self.__experiment_path, "runs",str(self.__experiment_config['current_run_number'])))
        self.save_experiment_config()
        shutil.copy(path.join(self.__experiment_path, "config.json"), path.join(self.__experiment_path,"runs",str(self.__experiment_config['current_run_number']), "config.json"))

        
        
        
        

    
        
            
                       
