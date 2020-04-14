from __future__ import division

import matplotlib as mpl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools, random, math
df = pd.read_csv('./arrhythmia.data', header=None, na_values="?")

# Replace each missing value with the mode
# The preferred pandas function for finding missing values is isnull()
for i in range(280):
    if df[i].isnull().sum() > 0:
        df.iloc[:,i].fillna(df[i].mode()[0], inplace=True)

##### Node class############################
class Node(object):
    def __init__(self):
        self.name = None
        self.node_type = None
        self.label = None
        self.data = None
        self.split = None
        self.children = []
        
    def __repr__(self):
        # Return the representation of the current node as a string
        # Write your code here
        
        data = self.data
        if self.node_type != 'leaf':
            s = (f"{self.name} Internal node with {data[data.columns[0]].count()} rows; split"
                f" {self.split.split_column} at {self.split.point:.2f} for children with"
                f" {[p[p.columns[0]].count() for p in self.split.partitions()]} rows"
                f" and infomation gain {self.split.info_gain:.5f}")
        else:
            s = "testing"
            s = (f"{self.name} Leaf with {data[data.columns[0]].count()} rows, and label"
                 f" {self.label}")
        return s

################ Split class##############################
class Split(object):
    def __init__(self, data, class_column, split_column, point=None):
        self.data = data
        self.class_column = class_column  ## the classes column
        self.split_column = split_column  ##  split for a node
        self.info_gain = None
        self.point = point         # spliting point
        self.partition_list = None # stores the data points on each side of the split
        self.compute_info_gain()
        
    def compute_info_gain(self):
        
        # sort by class_column
        self.data = self.data.sort_values(by= self.split_column) # dataframe sorted by split_column
        #class array
        class_label_array = self.data[self.class_column]
        # split array
        split_label_array = self.data[self.split_column]
        # count the number for first split-- same as the parent entropy
        C = self.data[self.class_column].value_counts() # count for each unique label sort by label
        C = C.sort_index()
        K = list(set(self.data[self.class_column])) # label
        K.sort()
        Z = [0 for i in range(len(K))]  # 0 count
        prev_gt = dict(zip(K,C))       # greater
        prev_le = dict(zip(K,Z))       # smaller
        ## create two dictionary with spilit1:count1......
        Ep = self.entropy(prev_gt, prev_le) # parent entropy
        G0 = 0   # for first split, info_gain is 0
        G_max = 0 # max gain
        # an np array of unique split values
        ori_val = split_label_array.unique()
        ## this return a numpy array
        if len(ori_val) == 2:
            # calculate binary gain
            ## format: classlabel:count      
            zero_dict = dict(self.data[self.data[self.split_column]==0][self.class_column].value_counts())
            one_dict = dict(self.data[self.data[self.split_column]!=0][self.class_column].value_counts())
                  
            entropy = self.entropy(zero_dict, one_dict)
            gain = Ep - entropy
            self.info_gain = gain
            self.point = 0.5

        # get the mid point list unique
        ori_val2 = np.insert(ori_val, 0, ori_val[0]-1) # copy of ori_val with two first element
        mid_val = [(ori_val[i]+ori_val2[i])/2 for i in range(len(ori_val))] # [1,3,4,6]
        mid_val = np.insert(mid_val, len(mid_val), mid_val[-1]+1)

        # count for continuous info_gain
        for i in range(len(mid_val)-1):
            # get previous splitting value
            pre_split_v = ori_val[i]
            # traverse self.data, change cooresponding dict by the class value
#             if i == 1:
#                 G_cur = Ep - self.entropy(prev_gt, prev_le)
#             else:
#                 prev_le_temp = dict(self.data[self.data[self.split_column] < pre_split_v][self.class_column].value_counts())
#                 prev_gt_temp = dict(self.data[self.data[self.split_column] >= pre_split_v][self.class_column].value_counts())
            temp_list = class_label_array[split_label_array == ori_val[i]].values
            for index in temp_list:
                prev_gt[index] -=1
                prev_le[index] +=1
                G_cur = Ep - self.entropy(prev_gt, prev_le)
            if G_cur > G_max:
                G_max = G_cur
                self.point = mid_val[i+1]
        self.info_gain = G_max

    
    # d1:dict({classLabel: count})
    def entropy(self, d1, d2):
        
        # weight for d1
        sum_d1 = sum(d1.values())
        sum_d2 = sum(d2.values())
        # n+p
        sum_np = sum_d1 + sum_d2
        S1 = S2 = 0
        if(sum_d1):
            for v in d1.values():
                if(v):
                    # compute the entropy value
                    S1 += (-(v/(sum_d1))*np.log2(v/(sum_d1)))
            S1 *= sum_d1/sum_np
            ## weighted average
        if(sum_d2):
            for v in d2.values():
                if(v):
                    # compute the entropy value
                    S2 += (-(v/(sum_d2))*np.log2(v/(sum_d2)))
            S2 *= sum_d2/sum_np
            ## weighted average
        return S1+S2
                    
                    
    def partitions(self):
        '''Get the two partitions (child nodes) for this split.'''
        if self.partition_list:
            # This check ensures that the list is computed at most once.  Once computed 
            # it is stored
            return self.partition_list
        data = self.data
        split_column = self.split_column
        partition_list = []
        partition_list.append(data[data[split_column] <= self.point])
        partition_list.append(data[data[split_column] > self.point])
        return partition_list
    

print("check the split and node class by print a specific split point:")
print("split point of column 100 is :")
print(Split(df,279,split_column=100).point)

#############Decision Tree class##################################

class DecisionTree(object):

    def __init__(self, max_depth=None):
        if (max_depth is not None and (max_depth != int(max_depth) or max_depth < 0)):
            raise Exception("Invalid max depth value.")
        self.max_depth = max_depth
        

    def fit(self, data, class_column):
        '''Fit a tree on data, in which class_column is the target.'''
        if (not isinstance(data, pd.DataFrame) or class_column not in data.columns):
            raise Exception("Invalid input")
            
        self.data = data
        self.class_column = class_column
        self.non_class_columns = [c for c in data.columns if 
                                  c != class_column]
        
        self.root = self.recursive_build_tree(data, depth=0, name='0')
            
    def recursive_build_tree(self, data, depth, name):
        
        node = Node()
        node.name = name
        node.data = data

        # base case, when current depth is equal to the maxmimum level
        # create a leaf node and return it
        if depth == self.max_depth:
            node.node_type = 'leaf'
            node.label = node.data[self.class_column].mode()[0]
            return node
            
        # otherwise, find the maximum split among columns
        # basically, you want to iterate through feature columns to find the best column for splitting
        split_column = list(data.columns)
        
        max_info_gain_column = 0
        max_info_gain = -float("inf")
        for i in range(len(split_column)-2):
            split = Split(data, self.class_column,split_column[i])
            # update max_info_gain and max_info_gain_column
            if split.info_gain > max_info_gain:
                max_info_gain = split.info_gain
                max_info_gain_column = split_column[i]
            
        # after finding the max_info_gain, actually do splitting
        split = Split(data, self.class_column, max_info_gain_column)
        node.split = split
        new_data = split.partitions()
        ## delete the column if already split on it
        droped = [c for c in data.columns if c != max_info_gain_column]
        
        left_data = (new_data[0][droped])
        right_data = (new_data[1][droped])
        ## if one child has no examples, stop diverging
        if left_data.empty or right_data.empty:
            node.node_type = 'leaf'
            node.label = node.data[self.class_column].mode()[0]
        else:
            node.children.append(self.recursive_build_tree(left_data, depth+1, name+'.0'))
            node.children.append(self.recursive_build_tree(right_data, depth+1, name+'.1'))
        return node

    
    def predict(self, test):
        # make predictions on the test set
        predictions = []
        for i in range(len(test)):
            cur_data = test.iloc[[i]] # row data at index i 
            pred_y = self.predict_y(self.root, cur_data) # recursively traverse the decision_tree
            predictions.append(pred_y)
            
        return predictions
            
    def predict_y(self, node, cur_data):
        if node.node_type == 'leaf':
            return node.label
        
        split = node.split
        split_val = split.point
        ## this is the threshould
        col = split.split_column
        my_val = cur_data[col] 
        my_val = my_val.iloc[0]
        if my_val< split_val:
            return self.predict_y(node.children[0],cur_data)
        else:
            return self.predict_y(node.children[1],cur_data)
        
                
    def print(self):
        self.recursive_print(self.root)
    
    def recursive_print(self, node):
        print(node)
        for u in node.children:
            self.recursive_print(u)

print("test for decision tree class by printing a decision tree using small data: ")
tree = DecisionTree(3)
all_column = list(df.columns)
selected_column = random.sample(all_column[:-1],50)
small_data = df.iloc[:, selected_column + [len(all_column) - 1]]
tree.fit(small_data,279)
tree.print()

################### cross validation###########################
def validation_curve(df):
    # randomly shuffle the dataset rows 
    data = df.sample(frac=1)
   
    # replace the missing data with mode in that col
    split_column = list(data.columns)
    for i in split_column:
        if sum(data[i].isnull()):
            data[i] = data[i].fillna(data[i].mode())
    
    # divide the examples roughly into three partitions
    x = int(len(data)*1/3)
    data_1 = data.iloc[0:x, :]
    data_2 = data.iloc[x:2*x,:]
    data_3 = data.iloc[2*x:len(data),:]
    data_set = [data_1, data_2, data_3]
    
    print("start print the curve: ......")
    train_scores_mean = []
    test_scores_mean = []
    for max_depth in range(2,17,2): # When debugging, you may consider less depth
        train_accs = []
        test_accs = []
        tree = DecisionTree(max_depth)
        for i in range(3):
            
            # use 2/3 data as training set, and 1/3 as test set. 
            # train decision tree with max_depth
            # compute accuracy and append to the list
            training_set = pd.concat([data_set[j] for j in range(len(data_set)) if j != i])
            testing_set = pd.DataFrame(data_set[i])
            tree.fit(training_set,279)
            train_res = tree.predict(training_set)
            test_res = tree.predict(testing_set)
            train_acc = np.sum(train_res == training_set[279])/len(training_set)
            test_acc = np.sum(test_res == testing_set[279])/len(testing_set)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
        train_scores_mean.append(np.sum(train_accs)/3) 
        test_scores_mean.append(np.sum(test_accs)/3)        
        print(train_scores_mean,test_scores_mean)
       
    # plot one point on curve
    param_range = range(2,17,2)
    plt.title("Validation Curve of DecisionTree")
    plt.xlabel("Max depth")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    #  - train_scores_std
    plt.fill_between(param_range, train_scores_mean,
                     train_scores_mean, alpha=1,
                     color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean,
                     test_scores_mean , alpha=1,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()
    
validation_curve(df)