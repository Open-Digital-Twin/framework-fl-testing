import os
import ujson
import numpy as np
from sklearn.model_selection import train_test_split
batch_size = 10
train_size = 0.75 # merge original training set and test set, then split it manually. 
least_samples = batch_size / (1-train_size) # least samples for each client
alpha = 0.1 # for Dirichlet distribution

class Dataset:
    __least_samples: float
    batch_size: int
    train_size: float
    class_per_client: int 
    alpha: float
    config_path: str
    train_path: str
    test_path: str
    num_clients: int
    num_classes: int
    niid: int
    balance: bool
    partition: str
    

    def __init__(self, config_path: str, train_path: str, test_path: str, num_clients: int, num_classes: int,batch_size: int = 10, class_per_client: int=2, alpha: float = 0.1, niid: bool=False, 
        balance: bool=True, partition: str=None):
        self.batch_size= batch_size
        self.train_size= train_size
        self.__least_samples = batch_size / (1-train_size)
        self.niid = niid
        if not self.niid:
            self.partition = 'pat'
            self.class_per_client = num_classes
        else:
            self.partition = partition
            if partition is 'dir':
                self.alpha=alpha
                self.class_per_client = None
            else:
                self.class_per_client = class_per_client
                self.alpha=None
        self.config_path = config_path
        self.train_path = train_path
        self.test_path = test_path
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.balance = balance


    def check(self):
        # check existing dataset
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = ujson.load(f)
            if config['num_clients'] == self.num_clients and \
                config['num_classes'] == self.num_classes and \
                config['class_per_client'] == self.class_per_client and \
                config['non_iid'] == self.niid and \
                config['balance'] == self.balance and \
                config['partition'] == self.partition and \
                config['alpha'] == self.alpha and \
                config['batch_size'] == self.batch_size:
                #print("\nDataset already generated.\n")
                return True

        dir_path = os.path.dirname(self.train_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        dir_path = os.path.dirname(self.test_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        return False

class Cifar10Dataset(Dataset):
    self.

def separate_data(dataset: Dataset) -> bool:
    X = [[] for _ in range(dataset.num_clients)]
    y = [[] for _ in range(dataset.num_clients)]
    statistic = [[] for _ in range(dataset.num_clients)]

    dataset_content, dataset_label = data

    dataidx_map = {}

   

    if dataset.partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
                selected_clients = selected_clients[:int(num_clients/num_classes*class_per_client)]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients-1)]
            else:
                num_samples = np.random.randint(max(num_per/10, least_samples/num_classes), num_per, num_selected_clients-1).tolist()
            num_samples.append(num_all_samples-sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx+num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx+num_sample], axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        while min_size < least_samples:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    # assign data
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))
            

    del data
    # gc.collect()

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic


def split_data(X, y):
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train':[], 'test':[]}

    for i in range(len(y)):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_size, shuffle=True)

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

def save_file(config_path, train_path, test_path, train_data, test_data, num_clients, 
                num_classes, statistic, niid=False, balance=True, partition=None):
    config = {
        'num_clients': num_clients, 
        'num_classes': num_classes, 
        'non_iid': niid, 
        'balance': balance, 
        'partition': partition, 
        'Size of samples for labels in clients': statistic, 
        'alpha': alpha, 
        'batch_size': batch_size, 
    }

    # gc.collect()
    print("Saving to disk.\n")

    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")