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
import nets
import loader
import loss

# Here we give a demo for MLP+SVD on MNIST, other experiments can be done set differenct models and dataset.
# All models and datasets are involved in this repo.

model = Net(threshold=3)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,60,70,80], gamma=0.2)

fmt = {'tr_los': '3.1e', 'te_loss': '3.1e', 'sp_0': '.3f', 'sp_1': '.3f', 'lr': '3.1e', 'kl': '.2f'}
logger = Logger('sparse_vd', fmt=fmt)

train_loader, test_loader = get_mnist(batch_size=100)
sgvlb = SGVLB(model, len(train_loader.dataset))

kl_weight = 0.02
epochs = 100

for epoch in range(1, epochs + 1):
    start = time.time()
    scheduler.step()
    model.train()
    train_loss, train_acc = 0, 0 
    kl_weight = min(kl_weight+0.02, 1)
    logger.add_scalar(epoch, 'kl', kl_weight)
    logger.add_scalar(epoch, 'lr', scheduler.get_lr()[0])
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data
        target = target
        data = data.view(-1, 28*28)
        optimizer.zero_grad()
        
        output = model(data)
        pred = output.data.max(1)[1] 
        loss = sgvlb(output, target, kl_weight)
        loss.backward()
        optimizer.step()
        
        train_loss += loss 
        train_acc += np.sum(pred.cpu().numpy() == target.cpu().data.numpy())

    logger.add_scalar(epoch, 'tr_los', train_loss / len(train_loader.dataset))
    logger.add_scalar(epoch, 'tr_acc', train_acc / len(train_loader.dataset) * 100)
    
    
    model.eval()
    test_loss, test_acc = 0, 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data
        target = target
        data = data.view(-1, 28*28)
        output = model(data)
        test_loss += float(sgvlb(output, target, kl_weight))
        pred = output.data.max(1)[1] 
        test_acc += np.sum(pred.cpu().numpy() == target.cpu().data.numpy())
        
    logger.add_scalar(epoch, 'te_loss', test_loss / len(test_loader.dataset))
    logger.add_scalar(epoch, 'te_acc', test_acc / len(test_loader.dataset) * 100)
    
    for i, c in enumerate(model.children()):
        if hasattr(c, 'kl_reg'):
            logger.add_scalar(epoch, 'sp_%s' % i, (c.log_alpha.cpu().data.numpy() > model.threshold).mean())
    
    end = time.time()  
    logger.add_scalar(epoch, 'time', end - start)
            
    logger.iter_info()

all_w, kep_w = 0, 0

for c in model.children():
    kep_w += (c.log_alpha.cpu().data.numpy() < model.threshold).sum()
    all_w += c.log_alpha.cpu().data.numpy().size

print('keept weight ratio =', all_w/kep_w)

import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib import rcParams

rcParams['figure.figsize'] = 16, 4
rcParams['figure.dpi'] = 200


log_alpha = (model.fc1.log_alpha.cpu().detach().numpy() < 3).astype(np.float)
W = model.fc1.W.cpu().detach().numpy()

# Normalize color map
max_val = np.max(np.abs(log_alpha * W))
norm = mpl.colors.Normalize(vmin=-max_val,vmax=max_val)

plt.imshow(log_alpha * W, cmap='RdBu', interpolation=None, norm=norm)
plt.colorbar()

s = 0
from matplotlib import rcParams
rcParams['figure.figsize'] = 8, 5

z = np.zeros((28*15, 28*15))

for i in range(15):
    for j in range(15):
        s += 1
        z[i*28:(i+1)*28, j*28:(j+1)*28] =  np.abs((log_alpha * W)[s].reshape(28, 28))
        
plt.imshow(z, cmap='hot_r')
plt.colorbar()
plt.axis('off')