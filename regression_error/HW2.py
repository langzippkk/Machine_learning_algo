import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from collections import defaultdict
from collections import OrderedDict
from random import random
from random import sample
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold # import KFold
from sklearn.impute import SimpleImputer
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn import preprocessing
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 200)
### Question 3##################################################
print("#########Question 3############################")
##(b)Using a Python script determine the value of E[C] when p =
##0.7, m = 15, and k = 5.

p = 0.7
m = 15
k = 5
U = []
for _ in range(1000):
	if random()>p:
		U.append(0)
	else:
		U.append(1)
def kfold(U,p,m,k):
	result = 0
	for _ in range(1000):
		T = np.array(sample(U,m))
		kf = KFold(n_splits=k)
		accuracy = 0
		for train_index,test_index in kf.split(T):
			train = T[train_index]
			test = T[test_index]
			if np.sum(train ==1)>int(m/2):
				predict = 1
			elif np.sum(train ==1) ==int(m/2):
				predict = np.random.randint(2,size=1)[0]
			else:
				predict = 0
			accuracy+=(float(np.sum(test==predict))/len(test))
		result+=(accuracy/k)
	return(result/1000)
print("expectation is:")
print(kfold(U,p,m,k))

# Use the Python script to plot E[C] for m in the range 10 to 100,
# and p and k as above. What can you infer from the plot?
res = []
for m in range(10,100):
	res.append(kfold(U,p,m,k))
plt.plot(range(10,100),res)
plt.show()


###### Question 5: quadratic weighted kappa###########################
print("#########Question 5############################")
###############To test the function, I put into two vectors a and b################
a = [1,2,3,4,5,4,3,2,1,1]
b = [2,3,4,5,4,1,3,4,5,4]
def kappa(a,b):
	##genetate O , which is just the confusion matrix
	O = confusion_matrix(a, b)
	## generate E, which is the count of the vectors
	d1 = defaultdict(int)	
	d2 = defaultdict(int)	
	for ele in (a):
		d1[ele]+=1
	for ele in b:
		d2[ele]+=1
		if ele not in d1:
			d1[ele] = 0
	for ele in a:
		if ele not in d2:
			d1[ele] = 0
	d1 = OrderedDict(sorted(d1.items()))
	d2 =  OrderedDict(sorted(d2.items()))
	E = np.outer(list(d1.values()),list(d2.values()))
##################################################################
	## generate w,the weight matrix
	N = len(O)
	w = np.zeros((N,N))
	for i in range(len(w)):
		for j in range(len(w)):
			w[i][j] =((i-j)**2)

	## result
	temp1 = 0
	temp2 = 0
	E = E/E.sum()
	O = O/O.sum()
	for i in range(len(w)):
		for j in range(len(w)):
			temp1+=w[i][j]*O[i][j]
			temp2+=w[i][j]*E[i][j]

	kappa = (1 - (temp1/temp2))
	print(kappa)
	print("Is my function for kappa equal to the sklearn: ?")
	print((kappa -cohen_kappa_score(a,b,weights='quadratic')<1e-8))
	return kappa


kappa(a,b)


####################Question 6####################################
print("#########Question 6############################")
## (a) Load the training data from the train.csv ﬁle at the site
train = pd.read_csv('train.csv')
## random sample 10000 rows
train = train.sample(n = 10000) 



## (b)Imputer##########################
## I use the most frequent strategy here because it does not make sense to 
## take average of some of the categorical variables.
num = train.isnull().sum()
den = train.count()
res = (num/den)
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
columns = train.columns
index = train.index
train = pd.DataFrame(imp.fit_transform(train))
train.columns = columns
train.index = index


## (c) onehot enconding##################################
Y = pd.DataFrame(train['Response'],dtype=np.float64)
CATEGORICAL_COLUMNS = ["Product_Info_1", "Product_Info_2", "Product_Info_3", "Product_Info_5", "Product_Info_6",\
                       "Product_Info_7", "Employment_Info_2", "Employment_Info_3", "Employment_Info_5", "InsuredInfo_1",\
                       "InsuredInfo_2", "InsuredInfo_3", "InsuredInfo_4", "InsuredInfo_5", "InsuredInfo_6", "InsuredInfo_7",\
                       "Insurance_History_1", "Insurance_History_2", "Insurance_History_3", "Insurance_History_4", "Insurance_History_7",\
                       "Insurance_History_8", "Insurance_History_9", "Family_Hist_1", "Medical_History_2", "Medical_History_3",\
                       "Medical_History_4", "Medical_History_5", "Medical_History_6", "Medical_History_7", "Medical_History_8",\
                       "Medical_History_9", "Medical_History_11", "Medical_History_12", "Medical_History_13", "Medical_History_14",\
                       "Medical_History_16", "Medical_History_17", "Medical_History_18", "Medical_History_19", "Medical_History_20",\
                       "Medical_History_21", "Medical_History_22", "Medical_History_23", "Medical_History_25", "Medical_History_26",\
                       "Medical_History_27", "Medical_History_28", "Medical_History_29", "Medical_History_30", "Medical_History_31",\
                       "Medical_History_33", "Medical_History_34", "Medical_History_35", "Medical_History_36", "Medical_History_37",\
                       "Medical_History_38", "Medical_History_39", "Medical_History_40", "Medical_History_41"]

OTHER_COLUMNS = ["Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI",
                      "Employment_Info_1", "Employment_Info_4", "Employment_Info_6",
                      "Insurance_History_5", "Family_Hist_2", "Family_Hist_3", "Family_Hist_4", "Family_Hist_5","Medical_History_1", "Medical_History_10", "Medical_History_15", "Medical_History_24", "Medical_History_32"]

categorical = train[CATEGORICAL_COLUMNS].applymap(str)
categorical = pd.get_dummies(categorical, drop_first=True)
train =pd.concat([categorical,train[OTHER_COLUMNS]],axis = 1)



#####(d) decision tree: validation curve and learning curve
## Decision Tree
param_range = [2,4,6,8,10]
train_scores, test_scores = validation_curve(
    tree.DecisionTreeClassifier(class_weight='balanced'), train, Y, param_name="max_depth", cv=5, 
    param_range=param_range,scoring="accuracy")
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(param_range,train_scores_mean,'o-', color="r",
                 label="Training score")

plt.plot(param_range,test_scores_mean,'o-', color="g",
                 label="Training score")

plt.xlabel("depth", fontsize=18)
plt.ylabel("accuracy", fontsize=18)

plt.show()
train_sizes, train_scores, valid_scores = learning_curve(
    tree.DecisionTreeClassifier(max_depth = 8), train, Y,train_sizes=np.array([0.1, 0.33, 0.55, 0.78, 1.]),cv=5,scoring="accuracy")

train_scores1_mean = np.mean(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
plt.plot(train_sizes,train_scores1_mean,'o-', color="r",
                 label="Training score")
plt.plot(train_sizes,valid_scores_mean,'o-', color="g",
                 label="validation score")
plt.xlabel("size", fontsize=18)
plt.ylabel("accuracy", fontsize=18)

plt.show()


##(e) Logistic Regression
other = train[OTHER_COLUMNS]
cat = [c for c in train.columns if c not in other]
train = train[cat]
name = other.columns
index = other.index
other = pd.DataFrame(normalize(other, axis=0))
other.columns = name
other.index = index
train =pd.concat([other,train],axis = 1)
## plot 

param_range =range(1,1000,100)
train = preprocessing.scale(train)
train_scores, test_scores = validation_curve(
    LogisticRegression(penalty='l2',max_iter=100), train, Y.values.ravel(), param_name="C", cv=5, 
    param_range=param_range,scoring="accuracy")
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(param_range,train_scores_mean,'o-', color="r",
                 label="Training score")

plt.plot(param_range,test_scores_mean,'o-', color="g",
                 label="Training score")

plt.xlabel("C", fontsize=18)
plt.ylabel("accuracy", fontsize=18)

plt.show()
train_sizes, train_scores, valid_scores = learning_curve(
    LogisticRegression(penalty = 'l2',C = 400), train, Y,train_sizes=np.array([0.1, 0.33, 0.55, 0.78, 1.]),cv=5,scoring="accuracy")

train_scores1_mean = np.mean(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
plt.plot(train_sizes,train_scores1_mean,'o-', color="r",
                 label="Training score")
plt.plot(train_sizes,valid_scores_mean,'o-', color="g",
                 label="validation score")
plt.xlabel("size", fontsize=18)
plt.ylabel("accuracy", fontsize=18)

plt.show()