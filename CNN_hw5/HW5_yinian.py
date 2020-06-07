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



################################Question 3: CNN class with Numpy################
DEBUG = True

class relu:
    
    def function(self, X):
        '''Return elementwise relu values'''
        return(np.maximum(np.zeros(X.shape), X))
    
    def derivative(self, X):
        # An example of broadcasting
        return((X >= 0).astype(int))
        
class no_act:
    """Implement a no activation function and derivative"""
    
    def function(self, X):
        return(X)
    
    def derivative(self, X):
        return(np.ones(X.shape))

# Set of allowed/implemented activation functions
ACTIVATIONS = {'relu': relu,
               'no_act': no_act}    
    
class CNNLayer:
    """
    Implement a class that processes a single CNN layer.
    
    ---
    
    Additional two comments by Amitabh for Spring, 2020: 
    
    (1) I emphasized slightly different 
    notation in my class presentations; this notebook uses the older version.  
    For one, in class I have been using d(a_i) to denote (del L)/(del a_i).
    Also in class I have emphasized that the nodes/neurons correspond to filters,
    and the image/pixel values correspond to outputs when these filters are applied
    at particular receptive field.  This notebook, however, calls each application
    a "neuron".  So if an image corresponding to the jth layer consists of 
    500 x 500 x 3 pixels, the notebook imagines they are the outputs of 500 x 500 x 3 
    "neurons". Which ignores the fact that there are actually only 3 distinct neurons, 
    (corresponding two 3 filters), each of which has been applied 500 x 500 times on
    an input image.
    
    (2) This starter code uses the older idea that the activation function is part
    of the layer.  I intend to change this in future.
    
    ---
    
    Let i be the index on the neurons in the ith layer, and j be the index on the 
    neurons in the next outer layer.  (Following Russell-Norvig notation.) Implement 
    the following:
    
    0. __init__: Initalize filters.
    
    1. forward step: Input a_i values.  Output a_j values.  Make copies of a_i values 
       and in_j values since needed in backward_step and filter_gradient.
    
    2. backward_step: Input (del L)/(del a_j) values.  Output (del L)/(del a_i).
    
    3. filter_gradient: Input (del L)/(del a_j) values. Output (del L)/(del w_{ij}) values.
    
    4. update: Given learning rate, update filter weights. 
    
    """
    
    def __init__(self, n, filter_shape, activation='no_act', stride = 1):
        """
        Initialize filters.
        
        filter_shape is (height of filter, width of filter, depth of filter). Depth 
        of filter should match depth of the forward_step input X.
        """
        
        self.num_filters = n
        self.stride = stride
        self.filter_shape = filter_shape
        try:
            self.filter_height = filter_shape[0]
            self.filter_width = filter_shape[1]
            self.filter_depth = filter_shape[2]
        except:
            raise Exception(f'Unexpected filter shape {filter_shape}')
        try:
            # Create an object of the activation class
            self.activation = ACTIVATIONS[activation]() 
        except:
            raise Exception(f'Unknown activation: {activation}')
        self.filters = self.filters_init()
        self.biases = self.biases_init()
        self.num_examples = None 
        # Set num_of_examples during forward step, and use to verify
        # consistency during backward step.  Similarly the data height, 
        # width, and depth.
        self.data_height = None
        self.data_width = None
        self.data_depth = None
        self.data_with_pads = None
        self.in_j = None  # the in_j values for next layer.
        self.im2col = None
        
    def filters_init(self):
        return np.random.random((self.num_filters, self.filter_height,
                                 self.filter_width, self.filter_depth))
    
    def biases_init(self):
        return np.random.random(self.num_filters)
    
    def set_filters(self, filters, biases):
        """Set filters to given weights.
        
           Useful in debugging."""
        if filters.shape != (self.num_filters, self.filter_height,
                                 self.filter_width, self.filter_depth):
            raise Exception(f'Mismatched filter shapes: stored '
                            f'{self.num_filters} {self.filter_shape} vs '
                            f'{filters.shape}.')
        if biases.shape != (self.num_filters,):
            raise Exception((f'Mismatched biases: stored '
                             f'{self.num_filters} vs '
                             f'{biases.shape}.'))
        self.filters = filters.copy()
        self.biases = biases.copy()
        
    def forward_step(self, X, pad_height=0, pad_width=0):
        """
        Implement a forward step.
        
        X.shape is (number of examples, height of input, width of input, depth of input).
        """
        
        try:
            # Store shape values to verify consistency during backward step
            self.num_examples = X.shape[0]
            self.data_height = X.shape[1]
            self.data_width = X.shape[2]
            self.data_depth = X.shape[3]
        except:
            raise Exception(f'Unexpected data shape {X.shape}')
        if self.data_depth != self.filter_depth:
            raise Exception(f'Depth mismatch: filter depth {self.filter_depth}'
                            f' data depth {self.data_depth}')
        self.pad_height = pad_height
        self.pad_width = pad_width
        self.input_height = self.data_height + 2 * self.pad_height
        self.input_width = self.data_width + 2 * self.pad_width
        
        # Add pad to X.  Only add pads to the 1, 2 (ht, width) axes of X, 
        # not to the 0, 4 (num examples, depth) axes.
        # 'constant' implies 0 is added as pad.
        X = np.pad(X, ((0,0),(pad_height, pad_height), 
                      (pad_width, pad_width), (0,0)), 'constant')
        
        # Save a copy for computing filter_gradient
        self.a_i = X.copy()  #
        
        # Get height, width after padding
        height = X.shape[1]
        width = X.shape[2]

        # Don't include pad in formula because height includes it.
        output_height = ((height - self.filter_height)/self.stride + 1)
        output_width = ((width - self.filter_width)/self.stride + 1)    
        if (
            output_height != int(output_height) or 
            output_width != int(output_width)
        ):
            raise Exception(f"Filter doesn't fit: {output_height} x {output_width}")
        else:
            output_height = int(output_height)
            output_width = int(output_width)
            
        #####################################################################
        # There are two ways to convolve the filters with X.
        # 1. Using the im2col method described in Stanford 231 notes.
        # 2. Using NumPy's tensordot method.
        #
        # (1) requires more code.  (2) requires understanding how tensordot
        # works.  Most likely tensordot is more efficient.  To illustrate both,
        # in the code below data_tensor is constructed using (1) and 
        # new_data_tensor is constructed using (2).  You may use either.
            
        # Stanford's im2col method    
        # Construct filter tensor and add biases
        filter_tensor = self.filters.reshape(self.num_filters, -1)
        filter_tensor = np.hstack((self.biases.reshape((-1,1)), filter_tensor))
        # Construct the data tensor
        # The im2col_length does not include the bias terms
        # Biases are later added to both data and filter tensors
        im2col_length = self.filter_height * self.filter_width * self.filter_depth
        num_outputs = output_height * output_width
        data_tensor = np.empty((self.num_examples, num_outputs, im2col_length))
        for h in range(output_height):
            for w in range(output_width):
                hs = h * self.stride
                ws = w * self.stride
                data_tensor[:,h*output_width + w, :] = X[:,hs:hs+self.filter_height,
                                    ws:ws+self.filter_width,:].reshape(
                                        (self.num_examples,-1))  
        # add bias-coeffs to data tensor
        self.im2col = data_tensor.copy()
        data_tensor = np.concatenate((np.ones((self.num_examples, num_outputs, 1)),
                                 data_tensor), axis=2)
        
        ## the im2col result, will use in backward
        ## data_tensor: (num_examples,out_w*out_h,(C*kernel_h*kernel_w+1))
        ## filter_tensor: (num_filters, C*kernel_h*kernel_w+1)
        output_tensor = np.tensordot(data_tensor, filter_tensor, axes=([2],[1]))
        output_tensor = output_tensor.reshape(
            (self.num_examples,output_height,output_width,self.num_filters))
        
        
        
        # NumPy's tensordot based method
        new_output_tensor = np.empty((self.num_examples, output_height, 
                                      output_width, self.num_filters))
        for h in range(output_height):
            for w in range(output_width):
                hs = h * self.stride
                ws = w * self.stride
                new_output_tensor[:,h,w,:] = np.tensordot(
                                                X[:, # example
                                                  hs:hs+self.filter_height, # height
                                                  ws:ws+self.filter_width,  # width
                                                  : # depth
                                                ], 
                                                self.filters[:, # filter 
                                                             :, # height
                                                             :, # width
                                                             :  # depth
                                                ], 
                                                axes = ((1,2,3),(1,2,3))
                                              )
                # Add bias term
                new_output_tensor[:,h,w,:] = (new_output_tensor[:,h,w,:] + 
                                              self.biases)
        # Check both methods give the same answer
        assert np.array_equal(output_tensor, new_output_tensor)
                
        
        self.in_j = output_tensor.copy() # Used in backward_step.
        output_tensor = self.activation.function(output_tensor) # a_j values
        return(output_tensor)
      
    def backward_step(self, D):
        """
        Implement the backward step and return (del L)/(del a_i). 
        
        Given D=(del L)/(del a_j) values return (del L)/(del a_i) values.  
        D (delta) is of shape (number of examples, height of output (i.e., the 
        a_j values), width of output, depth of output).
        
        Note that in our tests we will assume that (del L)/(del a_i) has
        shape corresponding to the padded X used in forward_step, and not the
        unpadded X.  Strictly speaking, only the unpadded X gradient is needed
        for gradient descent, but here we ask you to compute the gradient for the 
        padded X.  All our tests below, and in our grading will make this assumption.
        
        YOU MAY ASSUME STRIDE IS SET TO 1.        
        """
         ##  A = self.in_j : (self.num_examples,output_height,output_width,self.num_filters)
        try:
            num_examples = D.shape[0]
            delta_height = D.shape[1]
            delta_width = D.shape[2]
            delta_depth = D.shape[3]
        except:
            raise Exception(f'Unexpected delta shape {D.shape}')
        if num_examples != self.num_examples:
            raise Exception(f'Number of examples changed from forward step: '
                             f'{self.num_examples} vs {num_examples}')
        if delta_depth != self.num_filters:
            raise Exception(f'Depth mismatch: number of filters {self.num_filters}' 
                            f' delta depth {delta_depth}')
        # Make a copy so that we can change it
        prev_delta = D.copy()
        if prev_delta.ndim != 4:
            raise Exception(f'Unexpected number of dimensions {D.ndim}')
        new_delta = None
        
        ####################################################################
        # WRITE YOUR CODE HERE
        #   D (delta) is of shape (number of examples, height of output (i.e., the a_j values), width of output, depth of output = #of filters)
        ## new_delta if of shape  :(number of examples, height of input, width of input, depth of input).
        ## self.filters : (num_filters, filter_height, filter_width, filter_depth=input_depth)
        prev_delta = self.activation.derivative(prev_delta)*prev_delta
        new_delta =  np.zeros((self.a_i.shape))
        for i in range(D.shape[0]):
            for h in range(D.shape[1]):
                for w in range(D.shape[2]):
                    for c in range(D.shape[3]):
                        new_delta[i,h:h+self.filter_height,w:w+self.filter_width,:] += (self.filters[c,:,:,:]*prev_delta[i,h,w,c])             
        return(new_delta)
    
    def filter_gradient(self, D):
        """
        Return the filter_gradient.
        
        D = (del L)/(del a_j) has shape (num_examples, height, width, depth=num_filters)
        The filter_gradient (del L)/(del w_{ij}) has shape (num_filters, filter_height, 
        filter_width, filter_depth=input_depth)
        
        YOU MAY ASSUME STRIDE IS SET TO 1.
        
        """
         
        if DEBUG and D.ndim != 4:
            raise Exception(f'D has {D.ndim} dimensions instead of 4.')
        # D depth should match number of filters
        D_depth = D.shape[3]
        if DEBUG:
            if D_depth != self.num_filters:
                raise Exception(f'D depth {D_depth} != num_filters'
                                f' {self.num_filters}')
            if D.shape[0] != self.num_examples:
                raise Exception(f'D num_examples {D.shape[0]} !='
                                f'num_examples {self.num_examples}')
        f_gradient = None
        ####################################################################
        # WRITE YOUR CODE HERE
        ## self.ai:(number of examples, height of input, width of input, depth of input).
        ## D: (num_examples, height, width,  D_depth)
        ## self.im2col: (num_examples,out_w*out_h,(C*kernel_h*kernel_w))
        ## output f_gradient: (num_filters, filter_height, filter_width, filter_depth=input_depth)
        f_gradient = np.zeros((D_depth,self.filter_height,self.filter_width,self.filter_depth))
        bias_temp = np.zeros((D_depth))
        for i in range(D.shape[0]):
            for c in range(D_depth):
                for h in range(D.shape[1]):
                    for w in range(D.shape[2]):
                        temp = self.im2col[i,(h*D.shape[0] + w),:].reshape(self.filter_depth,self.filter_height,self.filter_width)
                        f_gradient[c,:,:,:] += temp*D[i,h,w,c]
                        bias_temp[c] += D[i,h,w,c]
        ## f_gradient = np.concatenate((f_gradient,bias_temp), axis=0)
        return f_gradient

################################ MNIST################################
print("################ MNIST########################################")
dtype = torch.float32
NUM_TRAIN = 49000
# 1. put data into dataloader
dataset_prefix = "fashionmnist" 
num_classes = 100
train = np.load("./hw5_data/{}_train.npy".format(dataset_prefix))

train_labels = train[:, -1]
train = train[:, :-1].reshape(-1, 1, 28, 28)
tensor_x = torch.Tensor(train)
tensor_y = torch.Tensor(train_labels)
my_dataset = data.TensorDataset(tensor_x,tensor_y)
loader_train = DataLoader(my_dataset, batch_size=64,
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
loader_val = DataLoader(my_dataset, batch_size=64,
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))


test = np.load("./hw5_data/{}_test.npy".format(dataset_prefix))
test = test.reshape(-1, 1, 28, 28)
loader_test = DataLoader(test,batch_size = 32)

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
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(1568, num_classes)

        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


model = ConvNet()
optimizer = optim.Adam(model.parameters(), lr=0.003, betas = (0.9,0.999),eps=1e-09, weight_decay=0,amsgrad = False)
train(model, optimizer, epochs=10)
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
dataset.to_csv('MNIST.csv')


################################## CIFAR ###################################
print("################################## CIFAR ###################################")
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
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 =  nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(8192, 2048)
        self.fc1 = nn.Linear(2048, num_classes)

        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        return out


model = ConvNet()
optimizer = optim.Adam(model.parameters(), lr=0.0005, betas = (0.9,0.999),eps=1e-06, weight_decay=0,amsgrad = False)
##optimizer = optim.SGD(model.parameters(), lr = 0.0001, momentum=0.9, weight_decay=0)
train(model, optimizer, epochs=20)
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