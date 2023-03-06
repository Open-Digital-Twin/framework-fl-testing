
import ujson
import numpy as np
from sklearn.model_selection import train_test_split
from os import path, makedirs
from .dataset import Dataset, Cifar10Dataset
import numpy as np
from typing import Optional, Dict

class StorageManager():

    experiment_path: str
    data_distribution_path: str
    use_cache: bool
    


    def __init__(self, experiment_path, data_distribution_path: Optional[str] = None):
        self.experiment_path = experiment_path
        self.data_distribution_path = data_distribution_path

    def __check_experiment_path(self, file_name:  Optional[str] = None):
        return path.exists(self.experiment_path + file_name)
    
    def __check_data_distribution_path(self):
        return path.exists(self.data_distribution_path)
    
    def __initialize_experiment_path(self):
        if not self.__check_experiment_path():
            makedirs(self.experiment_path)
        if not self.__check_experiment_path("runs"):
            makedirs(self.experiment_path + "runs")
        if not self.__check_experiment_path("logs"):
            makedirs(self.experiment_path + "logs")
        if not self.__check_experiment_path("config.json"):
            with open(self.experiment_path + "config.json", 'a') as config_file:
                ujson.dump({"dataset": {}},config_file)

    def __load_current_experiment_config(self) -> Dict:
        with open(self.experiment_path  + "config.json", 'r') as f:
            config = ujson.load(f)
            return config
       
     def __check_current_experiment_data_distribution(self):
        # check existing dataset
        with self.__load_current_experiment_config() as config:
            if config['data_distribution_path']:
                self.data_distribution_path = config['data_distribution_path']
                if self.__check_data_distribution_path():
                with open(config['data_distribution_path'], 'r') as f2:
                    dataset_config = ujson.load(f)
                    if dataset_config['dataset'] ==  config["dataset"]:
                        return True
                    
                if config['dataset'] == dataset.get_config():
                    return True
                else:
                    return False
            else:
                return False    
            
                



    
    
    def __check_current_dataset_config(self, dataset: Dataset):
        # check existing dataset
        if self.__check_experiment_path('config.json'):
            with open(self.experiment_path  + "config.json", 'r') as f:
                config = ujson.load(f)
            if config['dataset'] == dataset.get_config():
                return True
            else:
                return False
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

    def __save_file(self, dataset: Dataset, statistic, train_data, test_data):
        config= {}
        config.update({'dataset': dataset.get_config()})
        config.update({'statistic': statistic})

        # gc.collect()
        print("Saving to disk.\n")
        dir_path = path.dirname(self.experiment_path + "train/")
        if not path.exists(dir_path):
            makedirs(dir_path)
        dir_path = path.dirname(self.experiment_path + "test/")
        if not path.exists(dir_path):
            makedirs(dir_path)

        for idx, train_dict in enumerate(train_data):
            with open(self.experiment_path + "train/" + str(idx) + '.npz', 'wb') as f:
                np.savez_compressed(f, data=train_dict)
        for idx, test_dict in enumerate(test_data):
            with open(self.experiment_path + "test/" + str(idx) + '.npz', 'wb') as f:
                np.savez_compressed(f, data=test_dict)
        with open(self.experiment_path + "config.json", 'w') as f:
            ujson.dump(config, f)

        print("Finish generating dataset.\n")

    def distribute(self, dataset: Dataset):
        if self.__check_current_dataset_config(dataset):
            print("message good")
            return True
        else:
            X, y, statistic = self.__separate_data(dataset)
            train_data, test_data = self.__split_data(X, y, dataset)
            self.__save_file(dataset,statistic, train_data, test_data, )
                        
datasetmanager = DatasetManager(experiment_path = "./experiment/")
dataset = Cifar10Dataset(10,niid=True,partition='pat', balance=False)
datasetmanager.distribute(dataset)