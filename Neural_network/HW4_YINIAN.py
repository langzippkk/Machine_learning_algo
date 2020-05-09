import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class NN():
    def __init__(self, nodes_array, activation):
        self.n = len(nodes_array)-1
        self.nodes_array = nodes_array
        self.activation = activation
        self.weights = []
        self.bias =  []
        self.linear_temp =[None]*(self.n+1)  ## should have length n+1
        self.nonlinear_temp = [None]*(self.n+1) ## should have length n+1
        self.dz = [None]*(self.n+1) ## dloss/dz
        
        self.total_res = []  ## loss list
        self.accuracy = []  ## accuracy list
        
    def plot(self):
        '''
        plot the accuracy and loss vs iteration
        '''
        plt.plot(self.total_res)
        plt.title("Loss vs epoch in activation using "+ str(self.activation))
        plt.show()
        plt.plot(self.accuracy)
        plt.title("accuracy vs epoch in activation using "+ str(self.activation))
        plt.show()
        
    def set_weights(self):
        '''
        set weight using xavier initialization
        '''
        for i in range(self.n):
            self.weights.append(self.xavier(self.nodes_array[i], self.nodes_array[i+1]))
            self.bias.append(self.xavier(1,self.nodes_array[i+1]))

    def xavier(self,in1,out1):
        temp = np.sqrt(6 / (in1 + out1))
        low = -temp
        return (temp - low) * np.random.random_sample((in1,out1)) + low
   
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def sigmoidd(self,x):
        res = self.sigmoid(x)*(1-self.sigmoid(x))
        return res
    
    def relu(self,Z):
        A = np.maximum(0,Z)  
        return A
    
    def relu_backward(self,dz):
        return 1. * (dz > 0)


    def fit(self,X,y,alpha,t,lambda1):
        '''
        Input: 
        X:training input, m by n
        y: result, m by 1
        alpha:learning rate
        t: epoch
        lambda1: regularization
        fit the data with 80% of the training data and validate with 20% of the training data
        '''
        X,X_validate,y,y_validate = train_test_split(X, y, test_size=0.8, random_state=1)
        m = len(X)
        n = len(X[0])
        self.set_weights()
        y = np.asarray(pd.get_dummies(y))
        y_validate =np.asarray(pd.get_dummies(y_validate))
        
        for epoch in range(t):
            res = 0
            for example in range(m):
                temp = np.asarray(X[example])
                y_temp = np.asarray(y[example,:])
                y_hat = self.forward(temp)
                self.backward(temp,y_temp)
                self.update(alpha,lambda1)
                loss_weight = np.sum([np.sum(np.square(i)) for i in self.weights])
                res +=np.sum((y_temp-y_hat)**2) + loss_weight*lambda1/2
                ##res +=np.sum((y_temp-y_hat)**2)
            self.total_res.append(res/m)
            if epoch %3 == 0:
                prediction_temp = self.predict(X_validate,y_validate)
                truth = (np.argmax(y_validate,axis =1))
                accuracy_temp = sum(truth == prediction_temp)/len(y_validate)
                self.accuracy.append(accuracy_temp)
                print("validation accuracy at epoch  "+str(epoch)+"is  "+str(accuracy_temp))
                print("the validation loss at epoch  "+str(epoch)+"is  "+str(res/m))

    def forward(self,x):
        '''
        Input: x, the training data
        pass forward throught the network
        nonlinear_temp(first = x) -> linear_temp(z) -> nonlinear_temp(a) ->....
        return the last output from nonlinear_layer, which is the predict probability
        output: predicted probability
        '''
        #self.linear_temp[0] = np.array(x)
        self.nonlinear_temp[0] = np.array(x)
        if self.activation == 'sigmoid':
            for layer in range(1,self.n):
                h_old = self.nonlinear_temp[layer-1]
                # print(h_old.shape,self.weights[layer-1].shape,self.bias[layer-1].shape)
                h_new = h_old@self.weights[layer-1]+self.bias[layer-1]
                a = self.sigmoid(h_new)
                self.linear_temp[layer] =(h_new)
                self.nonlinear_temp[layer] = a
        if self.activation == 'RELU':
            for layer in range(1,self.n):
                h_old = self.nonlinear_temp[layer-1]
                h_new = h_old@self.weights[layer-1]+self.bias[layer-1]
                a = self.relu(h_new)
                self.linear_temp[layer] =(h_new)
                self.nonlinear_temp[layer] = a   
        h_old = self.nonlinear_temp[self.n-1]
        h_new = h_old@self.weights[self.n-1]+self.bias[self.n-1]
        a = self.sigmoid(h_new)
        self.linear_temp[self.n] = h_new
        self.nonlinear_temp[self.n] = a
        return self.nonlinear_temp[-1]
            
        
    def backward(self,x,y):
        '''
        input: training data x and label y
        calculate dloss/dz, save in dz
        
        '''
        self.dz[-1] = (y-self.nonlinear_temp[-1])
        if self.activation == 'sigmoid':
            for layer in range(self.n-1,0,-1):
                z = self.linear_temp[layer]
                self.dz[layer] = (self.dz[layer+1]@self.weights[layer].T*self.sigmoidd(z))
        if self.activation == 'RELU':
            for layer in range(self.n-1,0,-1):
                z = self.linear_temp[layer]
                self.dz[layer] = (self.dz[layer+1]@self.weights[layer].T*self.relu_backward(z))
                 
                         
    def update(self,alpha,lambda1):
        '''
        Input: learning parameter alpha,
        regularization parameter labmda
        update with SGD
        db = dz[l]
        dw = dz[l]*linear_temp[l-1]
        
        '''
        for layer in range(self.n):
            self.weights[layer] += (alpha*self.nonlinear_temp[layer].T.reshape((self.nodes_array[layer],1))@self.dz[layer+1]) - alpha*lambda1*self.weights[layer]
            #self.weights[layer] += (alpha*self.nonlinear_temp[layer].T.reshape((self.nodes_array[layer],1))@self.dz[layer+1])
            self.bias[layer] += alpha*self.dz[layer+1]
    
    def get_weights(self):
        '''
        method to print the weight
        '''
        res = []
        for i in range(self.n-1):
            res.append(self.weights[i])
            res.append(self.bias[i])
        return res
    
    
    def predict(self,test_X,y):
        '''
        input: validation/test feature X and validation/test label y
        pass the test data through network
        output: predictions as an indicator variable 
        '''
        for layer in range(self.n):
            test_X  = self.sigmoid(test_X@self.weights[layer])
        res = np.ravel(np.argmax(test_X,axis = 1))
        return res
        

if __name__ == "__main__":
## load in data and normalize the data
    df = pd.read_csv('./train.csv')
    X = np.asmatrix(df.drop('label',axis=1))/255
    y = np.asarray(df['label'])
    print("testing different hyperparameters with just 10 percent of the output:....")
    print("RELU activation with learning rate 0.1, regularization parameter 0.0001")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=1)
    NN1 = NN([784,155,10],'RELU')
    NN1.fit(X_train,y_train,0.1,10,0.0001)
    NN1.plot()
    print("sigmoid activation with learning rate 0.1, regularization parameter 0.0001")
    NN2 = NN([784,155,10],'sigmoid')
    NN2.fit(X_train,y_train,0.1,10,0.0001)
    NN2.plot()

## fit final model with full dataset
    print("#####################################final model ###########################")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    NN_full = NN([784,155,10],'sigmoid')
    NN_full.fit(X_train,y_train,0.1,10,0.0001)
    NN_full.plot()
    prediction_temp = NN_full.predict(X_test,y_test)
    truth = y_test
    accuracy_temp = sum(truth == prediction_temp)/len(y_test)
    print("The final accuracy with learning rate 0.1, regularization parameter 0.0001, is  " +str(accuracy_temp))