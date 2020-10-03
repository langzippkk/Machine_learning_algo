import numpy as np
import os
import pandas as pd

## mainly use pandas
def compute_distance(point,centroid):
	return np.sqrt(np.sum(point-centroid)**2)

def initialize_centroids(data_points,k):
	res = []
	for i in range(k):
		res.append(data_points.sample().values)
	return res

def iteration(data_point,num_of_cluster,number_of_iteration):
	centroid = initialize_centroids(data_point,num_of_cluster)
	k = len(centroid)
	cluster = {}
	for i in range(k):
		cluster[i] = []
	## use defaultdict
	for iter in range((number_of_iteration)):
		for i in range(len(data_point)):
			dist = []
			## E-step, assign data to clusters according to their closest centroid
			data = data_points.loc[i].values
			for j in range(k):
				dist.append(compute_distance(data,centroid[j]))
			index_temp = np.argmin(dist)
			if iter == number_of_iteration-1:
				cluster[index_temp].append(data)
		## M step 
		## calculate new centroid using means
		for i in range(k):
			centroid[i] = np.mean(cluster[i])
	return cluster

if __name__ == "__main__":
    data_points = pd.read_csv('Kmean.csv')
    columns = data_points.columns
    for i in columns:
        data_points[i]= data_points[i].astype(float)
    total_iteration = 1
    number_of_cluster = 3
    res = iteration(data_points,number_of_cluster,total_iteration)
    print(res)
