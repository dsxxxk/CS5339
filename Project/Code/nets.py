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
import layers

# MLP+SparseVD
class MLPSVD(nn.Module):
    def __init__(self, threshold):
        super(MLPSVD, self).__init__()
        self.fc1 = LinearSVDO(28*28, 300, threshold)
        self.fc2 = LinearSVDO(300,  100, threshold)
        self.fc3 = LinearSVDO(100,  10, threshold)
        self.threshold=threshold

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

# AlexNetSVD
class AlexNetSVD(nn.Module):
    def __init__(self, threshold):
        super(AlexNetSVD, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.threshold
        self.classifier = nn.Sequential(
            LinearSVDO(2*2*256,  4096, threshold)
            nn.ReLU(inplace=True),
            LinearSVDO(4096, 4096, threshold),
            nn.ReLU(inplace=True),
            LinearSVDO(4096, 10, threshold),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x

class LeNetSVD(nn.Module):
    def __init__(self):
        super(LeNetSVD, self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1,6,5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,120,5,stride=1,padding=2),
            nn.ReLU(),
        )
        self.threshold = threshold
        self.classifier=nn.Sequential(
            LinearSVDO(7*7*120,120, threshold),
            nn.ReLU(),
            LinearSVDO(120, 84, threshold),
            nn.ReLU(),
            LinearSVDO(84,10, threshold),
            nn.Sigmoid(),
        )
    def forward(self, x):
        out=self.features(x)
        out=out.view(out.size(0),-1)
        out=self.classifier(out)
        return out

class LeNetBR(nn.Module):
    def __init__(self):
        super(LeNetBR, self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1,6,5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,120,5,stride=1,padding=2),
            nn.ReLU(),
        )
        self.classifier=nn.Sequential(
            nn.Linear(7*7*120,120),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(84,10),
            nn.Sigmoid(),
        )
    def forward(self, x):
        out=self.features(x)
        out=out.view(out.size(0),-1)
        out=self.classifier(out)
        return out
