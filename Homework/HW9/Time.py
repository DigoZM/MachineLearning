# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:16:42 2020

@author: digo
"""
import timeit


# Example of making predictions
from math import sqrt
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

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
new_scale = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
time = []
for k in k_vec:
    start = timeit.default_timer()
    print(k)
    for i in range(len(data_ev_m)):
#        print(i)
        prediction = predict_classification(data_train_m, data_ev_m[i], k)
    stop = timeit.default_timer()
    time.append(stop-start)
plt.plot(new_scale, time)
plt.title('Time in fuction of K')
plt.ylabel('Time')
plt.xlabel('K Neighbors in powers of 2')
plt.show()