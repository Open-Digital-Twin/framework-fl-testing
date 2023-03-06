
from torch import utils
from torchvision import transforms, datasets
from numpy import array as Array
from os import path, makedirs
from typing import Tuple, Dict


DEFAULT_BATCH_SIZE = 10
DEFAULT_TRAIN_SIZE = 0.75
DEFAULT_ALPHA_SIZE = 0.1
DEFAULT_CLASS_PER_CLIENT = 2
DEFAULT_NIID = False
DEFAULT_BALANCE = True
DEFAULT_PARTITION = None
DEFAULT_RAW_DATA_PATH = "./data/rawdata/"

class Dataset:
    name: str
    least_samples: float
    batch_size: int
    train_size: float
    class_per_client: int 
    alpha: float
    config_path: str
    num_clients: int
    num_classes: int
    niid: int
    balance: bool
    partition: str
    

    def __init__(self,  
                 name: str,
                 num_clients: int, 
                 num_classes: int,
                 rawdata_path: str,
                 batch_size: int = DEFAULT_BATCH_SIZE, 
                 class_per_client: int=DEFAULT_CLASS_PER_CLIENT, 
                 alpha: float = DEFAULT_ALPHA_SIZE, 
                 niid: bool=DEFAULT_NIID, 
                 balance: bool=DEFAULT_BALANCE, 
                 partition: str=DEFAULT_PARTITION, 
                 train_size: float = DEFAULT_TRAIN_SIZE):
        self.name = name
        self.batch_size= batch_size
        self.train_size= train_size
        self.least_samples = batch_size / (1-train_size)
        self.niid = niid
        if not self.niid:
            self.partition = 'pat'
            self.class_per_client = num_classes
        else:
            self.partition = partition
            if partition == 'dir':
                self.alpha=alpha
                self.class_per_client = None
            else:
                self.class_per_client = class_per_client
                self.alpha=None
        self.rawdata_path = rawdata_path
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.balance = balance
    
    def get_config(self) -> Dict:
        return {
                            'name': self.name,
                            'num_clients': self.num_clients, 
                            'num_classes': self.num_classes, 
                            'non_iid': self.niid, 
                            'balance': self.balance, 
                            'partition': self.partition, 
                            'class_per_client': self.class_per_client,  
                            'alpha': self.alpha, 
                            'batch_size': self.batch_size
                
               
        }
    

    def load_rawdata(self):
        raise NotImplementedError

class Cifar10Dataset(Dataset):



    def __init__(self, 
                 num_clients: int, 
                 rawdata_path: str = DEFAULT_RAW_DATA_PATH + 'cifar-10',
                 batch_size: int = DEFAULT_BATCH_SIZE, 
                 class_per_client: int=DEFAULT_CLASS_PER_CLIENT, 
                 alpha: float = DEFAULT_ALPHA_SIZE, 
                 niid: bool=DEFAULT_NIID, 
                 balance: bool=DEFAULT_BALANCE, 
                 partition: str=DEFAULT_PARTITION, 
                 train_size: float = DEFAULT_TRAIN_SIZE):
        super().__init__("Cifar-10",num_clients, 10, rawdata_path, batch_size, class_per_client, alpha, niid, balance, partition, train_size)




    def get_rawdata(self) -> Tuple:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if not path.exists(self.rawdata_path):
            makedirs(self.rawdata_path)
        
        trainset = datasets.CIFAR10(root=self.rawdata_path, train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root=self.rawdata_path, train=False, download=True, transform=transform)
        trainloader = utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
        testloader = utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)
        for _, train_data in enumerate(trainloader, 0):
            trainset.data, trainset.targets = train_data
        for _, test_data in enumerate(testloader, 0):
            testset.data, testset.targets = test_data
        

        dataset_image = []
        dataset_label = []

        dataset_image.extend(trainset.data.cpu().detach().numpy())
        dataset_image.extend(testset.data.cpu().detach().numpy())
        dataset_label.extend(trainset.targets.cpu().detach().numpy())
        dataset_label.extend(testset.targets.cpu().detach().numpy())
        dataset_image = Array(dataset_image)
        dataset_label = Array(dataset_label)

        return dataset_image, dataset_label
    
    


  
            



    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])