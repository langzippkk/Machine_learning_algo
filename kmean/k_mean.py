import numpy as np
import os
import pandas as pd
from collections import defaultdict


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
	## use defaultdict
	for iter in range((number_of_iteration)):
		cluster = defaultdict(list)
		for i in range(len(data_point)):
			dist = []
			## E-step, assign data to clusters according to their closest centroid
			data = data_points.loc[i].values
			for j in range(k):
				dist.append(compute_distance(data,centroid[j]))
			index_temp = np.argmin(dist)
			cluster[index_temp].append(data)
		## M step 
		## calculate new centroid using means
		for i in range(k):
			centroid[i] = np.mean(cluster[i])
		print(centroid)
	return cluster

if __name__ == "__main__":
    data_points = pd.read_csv('Kmean.csv')
    total_iteration = 5
    number_of_cluster = 2
    res = iteration(data_points,number_of_cluster,total_iteration)
    print(res)