import numpy as np
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
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
