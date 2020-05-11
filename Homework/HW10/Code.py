# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 00:07:54 2020

@author: digo
"""
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from math import sqrt


	
# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(1, len(row1)):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)
 
# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[0] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

#Reading File
##Train data
f_train = open('training_set.txt', 'r')
data = f_train.read()
f_train.close()
data_train = data.split()
data_train = map(float, data_train)
data_train_m = np.array(data_train)
x = len(data_train_m)/3
y = 3
data_train_m = data_train_m.reshape(x,y)
data_train_t = data_train_m.transpose()
half = len(data_train_m)/2
data_train_t0 = data_train_t[:,:half]
data_train_t1 = data_train_t[:,half:]
##Evaluation data
file_open = 'evaluation_set.txt'
#file_open = 'training_set.txt'
f_train = open(file_open, 'r')
data = f_train.read()
f_train.close()
data_ev = data.split()
data_ev = map(float, data_ev)
data_ev_m = np.array(data_ev)
x = len(data_ev_m)/3
y = 3
data_ev_m = data_ev_m.reshape(x,y)
data_ev_t = data_ev_m.transpose()
half = len(data_ev_m)/2
data_ev_t0 = data_ev_t[:,:half]
data_ev_t1 = data_ev_t[:,half:]

##Calculating Euclidean distance from each point from evaluation set
#distance_v = []
#aux = []
#for i in range (len(data_ev_m)):
#    for j in range(len(data_train_m)):
#        dist = euclidean_distance(data_ev_m[i][1:], data_train_m[j][1:])
#        aux.append(dist, data_train_m[j][0])
#
#    




# Test distance function
#dataset = [[2.7810836,2.550537003,0],
#	[1.465489372,2.362125076,0],
#	[3.396561688,4.400293529,0],
#	[1.38807019,1.850220317,0],
#	[3.06407232,3.005305973,0],
#	[7.627531214,2.759262235,1],
#	[5.332441248,2.088626775,1],
#	[6.922596716,1.77106367,1],
#	[8.675418651,-0.242068655,1],
#	[7.673756466,3.508563011,1]]
k_vec = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
#k_vec = [1, 2, 4, 8, 16, 32, 64, 100]
error_v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
print('starting')
for i in range(len(data_ev_m)):
    neighbors = get_neighbors(data_train_m, data_ev_m[i], k_vec[10])
    error_i = 0
    print(i)
    for num_neighbor in k_vec:
        output_values = [row[0] for row in neighbors[:num_neighbor]]
        prediction = max(set(output_values), key=output_values.count)
        if(prediction != data_ev_m[i][0]):
            error_v[error_i] += 1
        error_i += 1
error_rate = []
for i in range (len(error_v)):
    error_rate.append(float(error_v[i])/len(data_ev_m))
print('\n')
print('KNN error')
print(min(error_rate)*100.00)
#Plotting
#plt.plot(k_vec, error_rate)
#plt.title('Error Rate For -1 Overlap')
#plt.xlabel('K Neighbors')
#plt.ylabel('Error Rate')
#plt.show()
#write results in a file
#file_object  = open('results.txt', 'a')
#for i in range (len(k_vec)):
#    file_object.write('%d %.17f\n' % ( k_vec[i], error_rate[i]))
#file_object.close()

