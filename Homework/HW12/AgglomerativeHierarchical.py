# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:46:29 2020

@author: digo
"""

import matplotlib.pyplot as plt
import numpy as np
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
points = data_train_m[:,1:]


# create clusters
hc = AgglomerativeClustering(n_clusters=2, affinity = 'euclidean', linkage = 'ward')
# save clusters for chart
y_hc = hc.fit_predict(points)


#Calculate Error Rate
error = 0
for i in range(len(y_hc)):
    if(y_hc[i] != data_train_t[0][i]):
        error += 1
error_rate = float(error)/len(data_train_m)
print('Agglomerative Hierarchical Error:')
print(error_rate*100.00)
print((1-error_rate)*100.00)

plt.scatter(points[y_hc ==0,0], points[y_hc == 0,1], s=10, c='red')
plt.scatter(points[y_hc==1,0], points[y_hc == 1,1], s=10, c='blue')
plt.show()

# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(points, method='ward'))