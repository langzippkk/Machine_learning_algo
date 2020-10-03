from sklearn import datasets
from sklearn.metrics import accuracy_score
from collections import Counter
import numpy as np


def split(X,Y,test_size):
	split_point = int(len(X)*test_size)
	return X[:split_point],X[split_point:],Y[:split_point],Y[split_point:]


class KNN():
	def fit(self,X_train,Y_train,k):
		self.X_train = X_train
		self.Y_train = Y_train
		self.k = k

	def compute_distance(self,a,b):
		return np.sqrt(np.sum(a-b)**2)

	def nearestNeighbour(self,row):
		dist_list = []
		res = []
		for i in range(len(self.X_train)):
			dist_list.append(self.compute_distance(X_train[i],X_test))
		for j in range(self.k):
			temp = np.argmin(dist_list)
			res.append(self.Y_train[temp])
			dist_list.pop(temp)
		return Counter(res).most_common(1)[0][0]


	def predict(self,X_tests):
		predicitons = []
		for X_test in X_tests:
			label = self.nearestNeighbour(X_test)
			predicitons.append(label)
		return predicitons



if __name__ == "__main__":
	iris = datasets.load_iris()
	X = iris.data
	Y = iris.target
	X_train,X_test,Y_train,Y_test = split(X,Y,test_size=0.8)
	classifier = KNN()
	classifier.fit(X_train,Y_train,3)
	predicitons = classifier.predict(X_test)
	print(accuracy_score(Y_test,predicitons))
