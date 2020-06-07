import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.utils import data
import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np
import pandas as pd
import os
import math
dtype = torch.float32
NUM_TRAIN = 49000
transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
            ])
# 1. put data into dataloader
dataset_prefix = "cifar100" 
num_classes = 100
cifar10_train = np.load("./hw5_data/{}_train.npy".format(dataset_prefix))

train_labels = cifar10_train[:, -1]
cifar10_train = cifar10_train[:, :-1].reshape(-1, 3, 32, 32)

tensor_x = torch.Tensor(cifar10_train)
tensor_y = torch.Tensor(train_labels)
my_dataset = data.TensorDataset(tensor_x,tensor_y)
loader_train = DataLoader(my_dataset, batch_size=64,
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
loader_val = DataLoader(my_dataset, batch_size=64,
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))


cifar10_test = np.load("./hw5_data/{}_test.npy".format(dataset_prefix))
cifar10_test = cifar10_test.reshape(-1, 3, 32, 32)
loader_test = DataLoader(cifar10_test, batch_size=64)

#########################
USE_GPU = True

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# Constant to control how frequently we print train loss
print_every = 100
print('using device:', device)
best_acc = 0

def train(model, optimizer, epochs=1):
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t,(x,y) in enumerate(loader_train):
            # (1) put model to training mode
            model.train()
            # (2) move data to device, e.g. CPU or GPU
            x = x.to(device)
            y = y.to(device)
            # (3) forward and get loss
            outputs = model(x)
            criterion = nn.CrossEntropyLoss()
            y = torch.tensor(y, dtype=torch.long, device=device)
            loss = criterion(outputs, y)
            # (4) Zero out all of the gradients for the variables which the optimizer
            # will update.
            ## pyTorch accumulates the gradients on subsequent backward passes. This is convenient while training RNNs. 
            optimizer.zero_grad()
            # (5) the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()
            # (6)Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()
            if t % print_every == 0:
                print('Epoch %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                val(loader_val, model)
                print()

def val(loader, model):
    global best_acc
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for x, y in loader:
            # (1) move to device, e.g. CPU or GPU
            x = x.to(device)
            y = y.to(device)
            # (2) forward and calculate scores and predictions
            outputs = model(x)
            _,predictions = torch.max(outputs.data,1)
            # (2) accumulate num_correct and num_samples
            num_samples += y.size(0)
            num_correct += (predictions == y).sum().item()
            ## .item() method change from tensor to numbers
        acc = float(num_correct) / num_samples
        if acc > best_acc:
            # (4)Save best model on validation set for final test
            best_acc = acc
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

def test(loader, model):
    res = []
    global best_acc
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            x = torch.tensor(x,dtype=dtype,device=device)
            outputs = model(x)
            _,predictions = torch.max(outputs.data,1)
            res.append(predictions)
    res = torch.cat(res)
    return res



# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=100):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(2048, num_classes)

        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


model = ConvNet()
optimizer = optim.Adam(model.parameters(), lr=0.001, betas = (0.9,0.999),eps=1e-08, weight_decay=0,amsgrad = False)
train(model, optimizer, epochs=15)
torch.save(model.state_dict(), 'checkpoint.pth')
# load saved model to best_model for final testing
best_model = ConvNet()
state_dict = torch.load('checkpoint.pth')
best_model.load_state_dict(state_dict)
best_model.eval()
best_model.cuda()
##########################################################################
result = test(loader_test, best_model)
result = result.cpu().numpy()
dataset = pd.DataFrame({'Category': result})
dataset.index.name = 'Id'
# write CSV
dataset.to_csv('CIFAR.csv')
