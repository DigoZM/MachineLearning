# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:21:08 2020

@author: digo
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
# import KMeans
from sklearn.cluster import KMeans
# import hierarchical clustering libraries
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


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
points = data_train_m[:,1:]

#Implementing clustering using K-means
# create kmeans object
kmeans = KMeans(n_clusters=2)
# fit kmeans object to data
kmeans.fit(points)
# print location of clusters learned by kmeans object
#print(kmeans.cluster_centers_)
clusters = kmeans.cluster_centers_
# save new clusters for chart
y_km = kmeans.fit_predict(points)

#Calculate Error Rate k-Means
error = 0
for i in range(len(y_km)):
    if(y_km[i] != data_train_t[0][i]):
        error += 1
error_rate = float(error)/len(data_train_m)
print('K-Mean Error:')
print(min(error_rate*100.00, (1-error_rate)*100.00))

#Implementind Hierarchical
# create clusters
hc = AgglomerativeClustering(n_clusters=2, affinity = 'euclidean', linkage = 'ward')
# save clusters for chart
y_hc = hc.fit_predict(points)

#Calculate Error Rate Hierarchical
error = 0
for i in range(len(y_hc)):
    if(y_hc[i] != data_train_t[0][i]):
        error += 1
error_rate = float(error)/len(data_train_m)
print('Agglomerative Hierarchical Error:')
print(min(error_rate*100.00, (1-error_rate)*100.00))

#Computing Distances k-mean
mean0 = [np.mean(data_train_t0[1]), np.mean(data_train_t0[2])]
mean1 = [np.mean(data_train_t1[1]), np.mean(data_train_t1[2])]
distanceClass0 = min(distance.euclidean(clusters[0], mean0), distance.euclidean(clusters[0], mean1))
distanceClass1 = min(distance.euclidean(clusters[1], mean0), distance.euclidean(clusters[1], mean1))
print('Distances from K-mean:')
print(distanceClass0**2)
print(distanceClass1**2)


#Computing Distances Hierarchical
mean0Hc = [0,0]
mean1Hc = [0,0]
total0 = 0
total1 = 0.0
for i in range(len(y_hc)):
    if(y_hc[i] == 0):
        mean0Hc[0] += data_train_t[1][i]
        mean0Hc[1] += data_train_t[2][i]
        total0 += 1
    else:
        mean1Hc[0] += data_train_t[1][i]
        mean1Hc[1] += data_train_t[2][i]
        total1 += 1
mean0Hc = [float(mean0Hc[0])/total0, float(mean0Hc[1])/total0]
mean1Hc = [mean1Hc[0]/total1, mean1Hc[1]/total1]
distanceClass0 = min(distance.euclidean(mean0Hc, mean0), distance.euclidean(mean0Hc, mean1))
distanceClass1 = min(distance.euclidean(mean1Hc, mean0), distance.euclidean(mean1Hc, mean1))
print('Distances from Hierarchical:')
print(distanceClass0**2)
print(distanceClass1**2)

#Plot Results k-mean
plt.scatter(points[y_km ==0,0], points[y_km == 0,1], s=10, c='red')
plt.scatter(points[y_km ==1,0], points[y_km == 1,1], s=10, c='blue')
plt.scatter(clusters[0][0], clusters[0][1], c= 'black', s = 50)
plt.scatter(clusters[1][0], clusters[1][1], c= 'black', s = 50)
plt.title('K-Mean Clustering')
plt.show()

plt.scatter(points[y_hc ==0,0], points[y_hc == 0,1], s=10, c='red')
plt.scatter(points[y_hc==1,0], points[y_hc == 1,1], s=10, c='blue')
plt.scatter(mean0Hc[0], mean0Hc[1], c= 'black', s = 50)
plt.scatter(mean1Hc[0], mean1Hc[1], c= 'black', s = 50)
plt.title('Top Down Clustering')
plt.show()
