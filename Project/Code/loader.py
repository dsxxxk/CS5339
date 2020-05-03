import math
import torch
import time
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from local_logger import Logger
from torch.nn import Parameter
from torchvision import datasets, transforms

# Load cifar 10
def get_cifar10(batch_size):
    trsnform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
        transform=trsnform), batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, download=True,
        transform=trsnform), batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader, test_loader

# Load MNIST
def get_mnist(batch_size):
    trsnform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
        transform=trsnform), batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True,
        transform=trsnform), batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader, test_loader

# Randonly Labeling Cifar10
class CIFAR10RandomLabels(datasets.CIFAR10):
    def __init__(self, corrupt_prob=0.0, num_classes=10, **kwargs):
        super(CIFAR10RandomLabels, self).__init__(**kwargs)
        self.n_classes = num_classes
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)
    
    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.train_labels if self.train else self.test_labels)
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        labels = [int(x) for x in labels]
        
        if self.train:
            self.train_labels = labels
        else:
            self.test_labels = labels