# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:18:38 2020

@author: digo
"""

import matplotlib.pyplot as plt
import numpy as np
# import KMeans
from sklearn.cluster import KMeans


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
points = data_train_m[:,1:]

#Implementing clustering using K-means
# create kmeans object
kmeans = KMeans(n_clusters=2)
# fit kmeans object to data
kmeans.fit(points)
# print location of clusters learned by kmeans object
print(kmeans.cluster_centers_)
clusters = kmeans.cluster_centers_
# save new clusters for chart
y_km = kmeans.fit_predict(points)

#Calculate Error Rate
error = 0
for i in range(len(y_km)):
    if(y_km[i] != data_train_t[0][i]):
        error += 1
error_rate = float(error)/len(data_train_m)
print('K-Mean Error:')
print(error_rate*100.00)

#Plot Results
plt.scatter(points[y_km ==0,0], points[y_km == 0,1], s=10, c='red')
plt.scatter(clusters[0][0], clusters[0][1], c= 'black', s = 50)
plt.scatter(points[y_km ==1,0], points[y_km == 1,1], s=10, c='blue')
plt.scatter(clusters[1][0], clusters[1][1], c= 'black', s = 50)
plt.show()

