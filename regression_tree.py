import numpy as np
import math
MIN_ERROR = 1e-10

class Node:
    
    def __init__(self):
        self.isleaf = False
        self.yhat = None 
        self.split_dim = -1
        self.split_dim_value = None
        self.left_child = None
        self.right_child = None
        
    def make_leaf(self, yhat):
        self.isleaf = True
        self.yhat = yhat #yhat is the value predicted at this leaf
        
    def make_internal(self, split_dim, split_threshold, left_child, right_child):
        self.isleaf = False
        self.split_dim = split_dim
        self.split_threshold = split_threshold
        self.left_child = left_child
        self.right_child = right_child
        self.split_dim_value = split_threshold
        
    def print(self):
        if self.isleaf:
            print(f'Predict {self.yhat}')
        else:
            print(f'Split on dim {self.split_dim} at value {self.split_threshold}')

class RTree():
    
    def __init__(self):
        self.min_error = MIN_ERROR
    
    @staticmethod
    def get_error(y):
        '''Return the squared error'''
        if len(y) == 0:
            return 0
        else:
            return ((y-y.mean())**2).sum()
    
    @staticmethod
    def split_examples(X, y, dim, value):        
        left_X = X[X[:,dim] < value]
        left_y = y[X[:,dim] < value]
        right_X = X[X[:,dim] >= value]
        right_y = y[X[:,dim] >= value]
        return ((left_X, left_y), (right_X, right_y))
    
    def build_subtree(self, X, y):
        
        # Implement the following:
        #
        # (In the description below dimension refers to an input dimension. If there are
        # two inputs, x_1 and x_2, then there are two dimension values, 0 and 1.)
        #
        # 1. The subtree should be a leaf if the X values are not very different,
        #    or the squared error is at most the minimum_error.  If so, create a leaf, 
        #    and return it. 
        # 2. Else find the dimension and split threshold for that dimension such that the 
        #    total squared error for the child nodes is minimized. For split threshold choose
        #    points between distinct values in the training set for that dimension. Create
        #    an internal node corresponding to the split, and recursively call 
        #    build_subtree() on left and right examples.  To split_examples for a 
        #    potential split use provided function split_examples().  To determine
        #    error for a given set of examples use provided function get_error().
        #    Return the root of the created subtree.
                        
        if y is None or y.shape[0] == 0:
            raise Exception("Empty data", X, y)
        
        # Part (1) above
        Xm = X.mean(axis=0)
        if ((np.abs(X - Xm) <= self.min_error).all() or
                self.get_error(y) <= self.min_error):            
            node = Node()
            node.make_leaf(y.mean())
            return node

        ## find dimension and split point
        dim_temp = None
        split_temp = None
        min_error = math.inf
        n_examples = len(X) #this is the number of examples
        n_dim = len(X[0])  # this is the number of dimensions
        for i in range(n_dim):
            distinct = np.unique(X[:,i])
            for j in range(len(distinct)-1):
                split_mid = float(distinct[j+1]+distinct[j])/2  ## split value
                (left_X, left_y), (right_X, right_y) = self.split_examples(X, y, i, split_mid) ## get the split result
                temp = self.get_error(left_y)+ self.get_error(right_y)
                ## find the index corresponding to the min_error
                if temp <= min_error:
                    dim_temp = i
                    split_temp = split_mid
                    min_error = temp
        ## after get the split point, actually do the split
        (left_X_final, left_y_final), (right_X_final, right_y_final) = self.split_examples(X, y, dim_temp, split_temp)
        ## delete the feature that has been splitted
#         left_X_final = np.delete(left_X_final, dim_temp, axis=1)
#         right_X_final = np.delete(right_X_final, dim_temp, axis=1)  
        ## recursively build the tree
        node = Node()
        left_child = self.build_subtree(left_X_final,left_y_final)
        right_child = self.build_subtree(right_X_final,right_y_final)
        node.make_internal(dim_temp, split_temp, left_child, right_child)
        return node
    
    def print_subtree(self, node, level):
        indent = "   " * level
        print(indent, end='')
        node.print()
        if not node.isleaf:
            self.print_subtree(node.left_child, level+1)
            self.print_subtree(node.right_child, level+1)
            
    def print(self):
        self.print_subtree(self.root, level=0)
        
    def fit(self, X, y):
        self.root = self.build_subtree(X, y)
    
    def predict(self, X):
        # Return an np.array of predicted yhat values
        # for each example in X
        predictions = []
        for i in range(len(X)):
            cur_data = X[i]
            pred_y = self.predict_y(self.root,cur_data)
            predictions.append(pred_y)
        return predictions
        
    def predict_y(self,node,cur_data):
        if node.isleaf == True:
            return node.yhat
        ## internal node
        split = node.split_dim
        split_val = node.split_dim_value
        ## test value of certain dimension
        my_val = cur_data[split]
        ## recursively build
        if my_val <= split_val:
            return self.predict_y(node.left_child,cur_data)
        else:
            return self.predict_y(node.right_child,cur_data)
    
def test0():
    X = np.array([
        [1.0, 1],
        [5, 1],
        [1, 7],
        [6, 7]
    ])
    y = np.array([4, 4.5, 10, 10.7]).T
    rtree = RTree()
    rtree.fit(X, y)
    rtree.print()
    print(rtree.predict(X))
    assert (rtree.predict(X) == y).all()
    print('Test 0 passed')
    return rtree

tree0 = test0()     

def test1():
    X = np.array([
        [0.0, 0],
        [10, 0],
        [3, 7],
        [1, 8],
        [0, 10],
        [3, 7]
    ])
    y = np.array([40, 70, 45, 41, 40, 100]).T
    rtree = RTree()
    rtree.fit(X, y)
    rtree.print()
    print(rtree.predict(X))
    yhat = np.array([40, 70, 72.5, 41, 40, 72.5]).T
    assert (rtree.predict(X) == yhat).all()
    print('Test 1 passed')
    return rtree

tree1 = test1()

def test2():
    X = np.random.rand(1000, 20)
    y = np.random.rand(1000)
    rtree = RTree()
    rtree.fit(X, y)
    yhat = rtree.predict(X)
    max_error = np.abs(yhat-y).max()
    print(f'Max error {max_error}')
    print(f'Max error is usually 0, but depending on random X values may be as high as 1e-3.')
    return rtree

tree2 = test2()