import pdb
import random
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler

from .augmentation.randaugment import RandAugment
from .data_utils import get_sampler_by_name, get_data_loader, get_onehot, split_ssl_data
from .dataset import BasicDataset

import torchvision
from torchvision import datasets, transforms

mean, std = {}, {}
mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
mean['MLCC'] = [0.1778, 0.04714, 0.16583]

std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
std['cifar100'] = [x / 255 for x in [68.2,  65.4,  70.4]]
std['MLCC'] = [0.26870, 0.1002249, 0.273526]


def get_transform_cifar(mean, std, train=True):
    if train:
        return transforms.Compose([transforms.RandomHorizontalFlip(),
                                   transforms.RandomCrop(32, padding=4),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])
    else:
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])
def get_transform_MLCC(mean, std, train=True):
    if train:
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return data_transforms
    else:
        return transforms.Compose([
            # transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
                                     # transforms.Normalize(mean, std)])
        ])
# def get_transform(mean, std, train=True):
#     if train:
#         return transforms.Compose([transforms.RandomHorizontalFlip(),
#                                       transforms.RandomCrop(32, padding=4),
#                                       transforms.ToTensor(),
#                                       transforms.Normalize(mean, std)])
#     else:
#         return transforms.Compose([transforms.ToTensor(),
#                                      transforms.Normalize(mean, std)])


from PIL import Image

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class SSL_Dataset_MLCC:
    """
    SSL_Dataset class gets dataset (cifar10, cifar100) from torchvision.datasets,
    separates labeled and unlabeled data,
    and return BasicDataset: torch.utils.data.Dataset (see datasets.dataset.py)
    """
    def __init__(self,
                 name='MLCC', #
                 train=True,
                 num_classes=10,
                 data_dir='./data'): #
        """
        Args
            name: name of dataset in torchvision.datasets (cifar10, cifar100)
            train: True means the dataset is training dataset (default=True)
            num_classes: number of label classes
            data_dir: path of directory, where data is downloaed or stored.
        """
        
        self.name = name
        self.train = train
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.transform = get_transform_MLCC(mean[name], std[name], train)
        
    def get_data(self):
        """
        get_data returns data (images) and targets (labels)
        """
        if self.train == True:
            print("Train")
            data_dir = os.path.join(self.data_dir, 'Train')
            # data_dir = '/root/dataset2/Samsung_fixmatch/Train'
        else:
            print("Test")
            data_dir = os.path.join(self.data_dir, 'Test')
            # data_dir = '/root/dataset2/Samsung_fixmatch/Test'

        labeled_dir_list = sorted(os.listdir(data_dir))
        labeled_data_dict = {key: [] for key in range(len(labeled_dir_list))}
        data = []
        target = []
        for i, listname in enumerate(labeled_dir_list):
            print(listname)
            file_directory = os.path.join(data_dir, listname)
            file_names = os.listdir(file_directory)
            for filename in file_names:
                # if I want to make dict_type data per each key
                # self.labeled_data_dict[i].append(pil_loader(os.path.join(file_directory, filename)))
                # if I want to make list_type data for all dataset
                data.append(np.asarray(pil_loader(os.path.join(file_directory, filename))))
                target.append(i)
        random.Random(0).shuffle(data)
        random.Random(0).shuffle(target)
        # random.seed(0)
        # shuffle_data = random.shuffle(data)
        # shuffle_target = random.shuffle(target)
        data_np = np.array(data)
        target_np = np.array(target)

        return data_np, target_np

    
    def get_dset(self, use_strong_transform=False, 
                 strong_transform=None, onehot=False):
        """
        get_dset returns class BasicDataset, containing the returns of get_data.
        
        Args
            use_strong_tranform: If True, returned dataset generates a pair of weak and strong augmented images.
            strong_transform: list of strong_transform (augmentation) if use_strong_transform is True
            onehot: If True, the label is not integer, but one-hot vector.
        """
        print("get_dset_params:", use_strong_transform, strong_transform, onehot)
        data, targets = self.get_data()
        num_classes = self.num_classes
        transform = self.transform
        data_dir = self.data_dir
        
        return BasicDataset(data, targets, num_classes, transform, 
                            use_strong_transform, strong_transform, onehot)
    
    
    def get_ssl_dset(self, num_labels, index=None, include_lb_to_ulb=True,
                            use_strong_transform=True, strong_transform=None, 
                            onehot=False):
        """
        get_ssl_dset split training samples into labeled and unlabeled samples.
        The labeled data is balanced samples over classes.
        
        Args:
            num_labels: number of labeled data.
            index: If index of np.array is given, labeled data is not randomly sampled, but use index for sampling.
            include_lb_to_ulb: If True, consistency regularization is also computed for the labeled data.
            use_strong_transform: If True, unlabeld dataset returns weak & strong augmented image pair. 
                                  If False, unlabeled datasets returns only weak augmented image.
            strong_transform: list of strong transform (RandAugment in FixMatch)
            oenhot: If True, the target is converted into onehot vector.
            
        Returns:
            BasicDataset (for labeled data), BasicDataset (for unlabeld data)
        """
        print("get_dset_params:", use_strong_transform, strong_transform, onehot)

        data, targets = self.get_data()
        num_classes = self.num_classes
        transform = self.transform
        data_dir = self.data_dir
        '''
        data: (50000, 32, 32, 3)
        targets: (50000)
        num_labels : 4000
        num_classes : 10
        index : None
        include_lb_to_ulb: True
        '''
        lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(data, targets, 
                                                                    num_labels, num_classes, 
                                                                    index, include_lb_to_ulb)
        
        lb_dset = BasicDataset(lb_data, lb_targets, num_classes, 
                               transform, False, None, onehot)
        
        ulb_dset = BasicDataset(ulb_data, ulb_targets, num_classes, 
                               transform, use_strong_transform, strong_transform, onehot)
        
        return lb_dset, ulb_dset


class SSL_Dataset_CIFAR:
    """
    SSL_Dataset class gets dataset (cifar10, cifar100) from torchvision.datasets,
    separates labeled and unlabeled data,
    and return BasicDataset: torch.utils.data.Dataset (see datasets.dataset.py)
    """

    def __init__(self,
                 name='cifar10',  #
                 train=True,
                 num_classes=10,
                 data_dir='./data'):  #
        """
        Args
            name: name of dataset in torchvision.datasets (cifar10, cifar100)
            train: True means the dataset is training dataset (default=True)
            num_classes: number of label classes
            data_dir: path of directory, where data is downloaed or stored.
        """

        self.name = name
        self.train = train
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.transform = get_transform_cifar(mean[name], std[name], train)


    def get_data(self):
        """
        get_data returns data (images) and targets (labels)
        """
        # getattr는 torchvision.datasets라는 class의 attribute를 반환해준다.
        dset = getattr(torchvision.datasets, self.name.upper())
        dset = dset(self.data_dir, train=self.train, download=True)
        data, targets = dset.data, dset.targets
        print("train_data:", self.train, data.size)
        return data, targets

    def get_dset(self, use_strong_transform=False,
                 strong_transform=None, onehot=False):
        """
        get_dset returns class BasicDataset, containing the returns of get_data.

        Args
            use_strong_tranform: If True, returned dataset generates a pair of weak and strong augmented images.
            strong_transform: list of strong_transform (augmentation) if use_strong_transform is True
            onehot: If True, the label is not integer, but one-hot vector.
        """
        print("get_dset_params():".format(self.train), use_strong_transform, strong_transform, onehot)
        data, targets = self.get_data()
        num_classes = self.num_classes
        transform = self.transform
        data_dir = self.data_dir

        return BasicDataset(data, targets, num_classes, transform,
                            use_strong_transform, strong_transform, onehot)

    def get_ssl_dset(self, num_labels, index=None, include_lb_to_ulb=True,
                     use_strong_transform=True, strong_transform=None,
                     onehot=False):
        """
        get_ssl_dset split training samples into labeled and unlabeled samples.
        The labeled data is balanced samples over classes.

        Args:
            num_labels: number of labeled data.
            index: If index of np.array is given, labeled data is not randomly sampled, but use index for sampling.
            include_lb_to_ulb: If True, consistency regularization is also computed for the labeled data.
            use_strong_transform: If True, unlabeld dataset returns weak & strong augmented image pair.
                                  If False, unlabeled datasets returns only weak augmented image.
            strong_transform: list of strong transform (RandAugment in FixMatch)
            oenhot: If True, the target is converted into onehot vector.

        Returns:
            BasicDataset (for labeled data), BasicDataset (for unlabeld data)
        """
        print("get_dset_params:", use_strong_transform, strong_transform, onehot)

        data, targets = self.get_data()
        num_classes = self.num_classes
        transform = self.transform
        data_dir = self.data_dir
        '''
        data: (50000, 32, 32, 3)
        targets: (50000)
        num_labels : 4000
        num_classes : 10
        index : None
        include_lb_to_ulb: True
        '''
        lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(data, targets,
                                                                    num_labels, num_classes,
                                                                    index, include_lb_to_ulb)

        lb_dset = BasicDataset(lb_data, lb_targets, num_classes,
                               transform, False, None, onehot)

        ulb_dset = BasicDataset(ulb_data, ulb_targets, num_classes,
                                transform, use_strong_transform, strong_transform, onehot)

        return lb_dset, ulb_dset
