import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import warnings
import math
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

print("#######################Question_3##############################")
def Question_3():
	## 3(b)
	X = np.array([[1.5, 6], [2, 1], [3,4], [4.5, 4.5],[4,4],[3.5,2]])
	y = np.array([1,1,1,2,2,2])
	clf = SVC(kernel = 'linear',C= 500)
	clf.fit(X, y)
	plot_decision_regions(X,y,clf)
	plt.show()

	def distance(x):
		y = clf.decision_function(x.reshape(1, -1))[0]
		w_norm = np.linalg.norm(clf.coef_)
		dist = abs(y / w_norm)
		return dist
	distance1 = [distance(x) for x in X]
	closest_point = X[np.argmin(distance1)]
	closest_dis = distance(closest_point)

	print("the closest point is: "+str(closest_point)+"with distance : "+str(closest_dis))

	##part c
	print("problem3 (c)")
	X = np.array([[1.5, 6], [2, 1], [3,4], [4.5, 4.5],[4,4],[5,2]])
	y = np.array([1,1,1,2,2,2])
	clf = SVC(kernel = 'linear',C= 500)
	clf.fit(X, y)
	plot_decision_regions(X,y,clf)
	plt.show()

	def distance(x):
		y = clf.decision_function(x.reshape(1, -1))[0]
		w_norm = np.linalg.norm(clf.coef_)
		dist = abs(y / w_norm)
		return dist
	distance1 = [distance(x) for x in X]
	closest_point = X[np.argmin(distance1)]
	closest_dis = distance(closest_point)

	print("the closest point is: "+str(closest_point)+"with distance : "+str(closest_dis))


	##part d
	print("problem3 (d)")
	X = np.array([[1.5, 6], [2, 1], [3,4], [4.5, 4.5],[4,4],[3.5,2]])
	y = np.array([1,1,1,1,2,2])
	C_list = [0.1,1,10]
	for c in C_list:
		clf = SVC(kernel = 'linear',C= c)
		clf.fit(X, y)
		plot_decision_regions(X,y,clf)
		plt.show()

	##part e
	print("problem3 (e)")
	X = np.array([[1.5, 6], [2, 1], [3,4], [4.5, 4.5],[4,4],[3.5,2]])
	y = np.array([1,1,1,1,2,2])
	clf = SVC(kernel = 'rbf',C= 500)
	clf.fit(X, y)
	plot_decision_regions(X,y,clf)
	plt.show()

Question_3()


print("#####################Question 4#################################")
def Question_4():
	def f_func(x):
	        return 3.0 + 4.0 * x - 0.05 * x**2

	def generate_data(size=200):
	    X = np.sort(np.random.random(size) * 100)
	    y = f_func(X) + (np.random.random(size) - 0.5) * 50
	    return(X,y)

	class KNNRegressor(sklearn.base.RegressorMixin):
	    
	    def __init__(self, k):
	        self.k = k
	        super().__init__()
	        
	    def fit(self, X,y):
	        self.X = X
	        self.y = y
	        self.nn = NearestNeighbors(n_neighbors=self.k)
	        self.nn.fit(X.reshape(-1,1))
	        
	    def predict(self, T):
	        predictions = []
	        _, neighbors = self.nn.kneighbors(T)
	        regressor = LinearRegression()
	        for i in range(T.shape[0]):
	            regressor.fit(self.X[neighbors[i]], self.y[neighbors[i]])
	            predictions.append(regressor.predict([T[i]]))
	        return np.asarray(predictions)


	class LWRegressor(sklearn.base.RegressorMixin):
	    
	    def __init__(self,gamma):
	        self.gamma = gamma
	        super().__init__()
	        
	    def fit(self, X,y):
	        self.X = X
	        self.y = y
	        
	    def predict(self, T):
	        predictions = []
	        regressor = LinearRegression()
	        for i in range(T.shape[0]):
	            sample_weight = [np.exp(-self.gamma*self.Square_Euclidean(T[i],temp)) for temp in self.X]
	            regressor.fit(self.X, self.y,sample_weight)
	            predictions.append(regressor.predict([T[i]]))
	        return np.asarray(predictions)

	    def Square_Euclidean(self,x,y):
	        return (sum([(a - b) ** 2 for a, b in zip(x, y)]))
	np.random.seed(1)
	u = np.linspace(0,100,300)
	f = f_func(u)
	X, y = generate_data()
	LW_reg = LWRegressor(1/40)
	LW_reg.fit(X.reshape(-1,1), y)
	predictions = LW_reg.predict(u.reshape(-1,1))

	knn_reg = KNNRegressor(5)
	knn_reg.fit(X.reshape(-1,1), y)
	predictions_2 = knn_reg.predict(u.reshape(-1,1))
	plt.plot(u,f, 'r', label='underlying function')
	plt.scatter(X, y, s=10, color='b', alpha=0.5, label='data')
	plt.plot(u,predictions, color='g', label='LW linear regression')
	plt.plot(u,predictions_2, color='y', label='knn linear regression')
	plt.legend()
	plt.show()

	## cross validation
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)
	k_space = [5,15,20,25,30]
	test_mse = math.inf
	best = 0
	list_1 = []
	for k_temp in k_space:
		knn_reg = KNNRegressor(k_temp)
		knn_reg.fit(X_train.reshape(-1,1), y_train)
		predictions_2 = knn_reg.predict(X_test.reshape(-1,1))
		test_mse_temp = mean_squared_error(y_test, predictions_2)
		list_1.append(test_mse_temp)
		if test_mse_temp < test_mse:
			best = k_temp
			test_mse = test_mse_temp

	print("best MSE is : "+str(test_mse)+" with neighbors number: "+str(best))
	plt.plot(k_space,list_1)
	plt.xlabel('K neighbors')
	plt.ylabel('MSE')
	plt.show()


	#### tuning gamma 

	gamma_space = [1/5,1/10,1/20,1/40]
	test_mse = math.inf
	best = 0
	list_2 = []
	for gamma_temp in gamma_space:
		reg = LWRegressor(gamma_temp)
		reg.fit(X_train.reshape(-1,1), y_train)
		predictions = reg.predict(X_test.reshape(-1,1))
		test_mse_temp = mean_squared_error(y_test, predictions)
		list_2.append(test_mse_temp)
		if test_mse_temp < test_mse:
			best = gamma_temp
			test_mse = test_mse_temp

	print("best MSE is : "+str(test_mse)+" with gamma: "+str(best))
	plt.plot(gamma_space,list_2)
	plt.xlabel('gamma')
	plt.ylabel('MSE')
	plt.show()


Question_4()


print("###################################Question_6######################")
def Question_6():
	data = pd.read_csv('Heart.csv').drop('Unnamed: 0', axis=1)
	data= data.dropna(how = 'any')
	pd.set_option('display.max_rows', 500)
	X = data.loc[:, data.columns].copy().drop('AHD',axis=1)
	y = data.loc[:, 'AHD'].copy()
	le = LabelEncoder()
	## transform y
	le.fit(y)
	y = (le.transform(y))
	## transform X, missing data, NaN
	# print(X_new.isna().sum())
	# print(pd.isnull(X_new).sum())
	enc = OneHotEncoder(drop = 'first')
	enc_df = pd.DataFrame(enc.fit_transform(X[['ChestPain','Thal']]).toarray())
	temp = X.drop(['ChestPain','Thal'],axis=1)
	index = [i for i in range(len(temp))]
	enc_df.index = index
	temp.index = index
	X_final  = pd.concat([enc_df,temp],axis = 1)
	## test train split
	X_train, X_test, y_train, y_test =train_test_split(X_final, y,stratify = y,test_size = 0.20)
	bagging = []
	oob_bagging = []
	rf = []
	oob_rf = []
	for number in range(1,300):
	    clf_oob = BaggingClassifier(n_estimators=number,oob_score = True,random_state=0).fit(X_train, y_train)
	    clf_rf = RandomForestClassifier(n_estimators=number,max_depth=5,oob_score = True, random_state=0).fit(X_train, y_train)
	    bagging.append(1-clf_oob.score(X_test,y_test))
	    oob_bagging.append(1-clf_oob.oob_score_)
	    rf.append(1-clf_rf.score(X_test,y_test))
	    oob_rf.append(1-clf_rf.oob_score_)
	              

	plt.plot(bagging,color = 'g',label = 'bagging')
	plt.plot(oob_bagging,color = 'r',label = 'oob_bagging')
	plt.plot(rf,color = 'b',label = 'rf')
	plt.plot(oob_rf,color = 'y',label = 'oob_rf')
	plt.xlabel('number of trees')
	plt.legend()
	plt.show()

Question_6()

print("#########################Question_7#######################")
def Question_7():
	chi_square = 9.34
	train_data = pd.DataFrame(np.random.normal(0, 1, 2000))
	test_data = pd.DataFrame(np.random.normal(0, 1, 10000))
	for i in range(9):
	    train_temp = pd.DataFrame(np.random.normal(0, 1, 2000))
	    test_temp = pd.DataFrame(np.random.normal(0, 1, 10000))
	    train_data = pd.concat([train_data,train_temp],axis = 1)
	    test_data =  pd.concat([test_data,test_temp],axis = 1)
	sum1 = (train_data*train_data).sum(axis = 1)
	y_train = [1 if i>chi_square else -1 for i in sum1]
	sum2 = (test_data*test_data).sum(axis = 1)
	y_test = [1 if i>chi_square else -1 for i in sum2]

	test_list = []
	train_list = []
	for M in range(50):
	    W = np.ones(len(train_data)) / len(train_data)
	    alpha_list = []
	    classifier_list = []
	    for m in range(M):
	        tree = DecisionTreeClassifier(max_depth=3)
	        tree.fit(train_data,y_train, sample_weight=W)
	        prob = tree.predict(train_data)
	        mask = (prob != y_train)
	        err = np.sum(W[mask])/np.sum(W)
	        alpha = (np.log(1 - err) - np.log(err))
	        W[mask] = W[mask]*np.exp(alpha)
	        alpha_list.append(alpha)
	        classifier_list.append(tree)
	    N = len(test_data)
	    N1 = len(train_data)
	    result = np.zeros(N)
	    result1 = np.zeros(N1)
	    for alpha, tree in zip(alpha_list, classifier_list):
	        result += alpha*tree.predict(test_data)
	        result1 += alpha*tree.predict(train_data)
	    result = np.sign(result)
	    result1 = np.sign(result1)
	    train_error = np.sum(result1 == y_train)/len(y_train)
	    test_error = np.sum(result == y_test)/len(y_test)
	    test_list.append(test_error)
	    train_list.append(train_error)
	    ## actually the test_accuracy and train_accuracy
	plt.plot(test_list,color='g',label = 'test')
	plt.plot(train_list,color = 'r',label = 'train')
	print(test_list)
	plt.legend()
	plt.show()

###############Part 2#######################################
	chi_square = 9.34
	train_data = pd.DataFrame(np.random.normal(0, 1, 1000))
	test_data = pd.DataFrame(np.random.normal(0, 1, 5000))
	for i in range(9):
	    train_temp = pd.DataFrame(np.random.normal(0, 1, 1000))
	    test_temp = pd.DataFrame(np.random.normal(0, 1, 5000))
	    train_data = pd.concat([train_data,train_temp],axis = 1)
	    test_data =  pd.concat([test_data,test_temp],axis = 1)
	sum1 = (train_data*train_data).sum(axis = 1)
	y_train = [1 if i>chi_square else -1 for i in sum1]
	sum2 = (test_data*test_data).sum(axis = 1)
	y_test = [1 if i>chi_square else -1 for i in sum2]

	train_data_2 = pd.DataFrame(np.random.normal(0, 1, 1000))
	test_data_2 = pd.DataFrame(np.random.normal(0, 1, 5000))
	for i in range(9):
	    train_temp = pd.DataFrame(np.random.normal(0, 1, 1000))
	    test_temp = pd.DataFrame(np.random.normal(0, 1, 5000))
	    train_data_2 = pd.concat([train_data_2,train_temp],axis = 1)
	    test_data_2 =  pd.concat([test_data_2,test_temp],axis = 1)
	sum1 = (train_data_2*train_data_2).sum(axis = 1)
	y_train_2 = [1 if i>12 else -1 for i in sum1]
	sum2 = (test_data_2*test_data_2).sum(axis = 1)
	y_test_2 = [1 if i>12 else -1 for i in sum2]
	train_data = pd.concat([train_data,train_data_2],axis = 0)
	test_data = pd.concat([test_data,test_data_2],axis = 0)
	y_train = y_train+y_train_2
	y_test = y_test+(y_test_2)

	test_list = []
	train_list = []
	for M in range(50):
	    W = np.ones(len(train_data)) / len(train_data)
	    alpha_list = []
	    classifier_list = []
	    for m in range(M):
	        tree = DecisionTreeClassifier(max_depth=3)
	        tree.fit(train_data,y_train, sample_weight=W)
	        prob = tree.predict(train_data)
	        mask = (prob != y_train)
	        err = np.sum(W[mask])/np.sum(W)
	        alpha = (np.log(1 - err) - np.log(err))
	        W[mask] = W[mask]*np.exp(alpha)
	        alpha_list.append(alpha)
	        classifier_list.append(tree)
	    N = len(test_data)
	    N1 = len(train_data)
	    result = np.zeros(N)
	    result1 = np.zeros(N1)
	    for alpha, tree in zip(alpha_list, classifier_list):
	        result += alpha*tree.predict(test_data)
	        result1 += alpha*tree.predict(train_data)
	    result = np.sign(result)
	    result1 = np.sign(result1)
	    train_error = np.sum(result1 == y_train)/len(y_train)
	    test_error = np.sum(result == y_test)/len(y_test)
	    test_list.append(test_error)
	    train_list.append(train_error)
	## actually the test_accuracy and train_accuracy
	plt.plot(test_list,color = 'r',label='test')
	plt.plot(train_list,color = 'g',label='train')
	print(test_list)
	plt.legend()
	plt.show()

Question_7()