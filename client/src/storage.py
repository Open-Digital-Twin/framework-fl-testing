import numpy as np
from os import path
import torch
from logging import INFO
from flwr.common.logger import log, configure as configure_logger
from torch.utils.data import DataLoader


class StorageManager():

    __experiment_path: str
    __experiment_base_path: str
    __id: int
    __experiment_name: str
    __hostname: str
   


    def __init__(self, experiment_base_path:str, experiment_name: str, hostname: str, id: int):
        self.__experiment_base_path = path.join(experiment_base_path)
        self.__experiment_path = path.join(experiment_base_path, experiment_name)
        self.__experiment_name= path.join(experiment_name)
        self.__hostname = hostname
        self.__id = id

    
        

 
    def initialize_experiment_path(self):
      
        file = path.join(self.__experiment_path,".temp","logs",f"{self.__hostname}.log")
        with open(file, mode='w+') as files:
            pass
        configure_logger("file", filename=file)
       
        log(INFO, f"Initializing experiment path: {self.__experiment_path}")

    
   




    def __read_data_(self, is_train=True):
        if is_train:
            train_data_dir = path.join(self.__experiment_path, 'data', 'train/')

            train_file = train_data_dir + str(self.__id) + '.npz'
            with open(train_file, 'rb') as f:
                train_data = np.load(f, allow_pickle=True)['data'].tolist()

            return train_data

        else:
            test_data_dir = path.join(self.__experiment_path, 'data', 'test/')

            test_file = test_data_dir + str(self.__id) + '.npz'
            with open(test_file, 'rb') as f:
                test_data = np.load(f, allow_pickle=True)['data'].tolist()

            return test_data
    
    """def read_client_data_text(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train, X_train_lens = list(zip(*train_data['x']))
        y_train = train_data['y']

        X_train = torch.Tensor(X_train).type(torch.int64)
        X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test, X_test_lens = list(zip(*test_data['x']))
        y_test = test_data['y']

        X_test = torch.Tensor(X_test).type(torch.int64)
        X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)

        test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
        return test_data"""


    def __read_data(self, is_train=True):
        #if dataset[:2] == "ag" or dataset[:2] == "SS":
        #    return self.read_client_data_text(dataset, idx)

        if is_train:
            train_data = self.__read_data_(is_train)
            X_train = torch.Tensor(train_data['x']).type(torch.float32)
            y_train = torch.Tensor(train_data['y']).type(torch.int64)

            train_data = [(x, y) for x, y in zip(X_train, y_train)]
            return train_data
        else:
            test_data = self.__read_data_(is_train)
            X_test = torch.Tensor(test_data['x']).type(torch.float32)
            y_test = torch.Tensor(test_data['y']).type(torch.int64)
            test_data = [(x, y) for x, y in zip(X_test, y_test)]
            return test_data
        
    def load_data(self, batch_size: int=32):
        train_data = self.__read_data(is_train=True)
        test_data = self.__read_data(is_train=False)
        num_examples = {"trainset" : len(train_data), "testset" : len(test_data)}


        return DataLoader(train_data, batch_size, drop_last=True, shuffle=False), DataLoader(test_data, batch_size, drop_last=False, shuffle=False), num_examples

 

