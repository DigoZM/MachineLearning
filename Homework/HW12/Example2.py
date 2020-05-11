# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:39:28 2020

@author: digo
"""

# import hierarchical clustering libraries
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


# create blobs
data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.6, random_state=50)
# create np array for data points
points = data[0]
# create scatter plot
plt.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap='viridis')
plt.xlim(-15,15)
plt.ylim(-15,15)
plt.show()



# create clusters
hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'ward')
# save clusters for chart
y_hc = hc.fit_predict(points)

plt.scatter(points[y_hc ==0,0], points[y_hc == 0,1], s=10, c='red')
plt.scatter(points[y_hc==1,0], points[y_hc == 1,1], s=10, c='black')
plt.scatter(points[y_hc ==2,0], points[y_hc == 2,1], s=10, c='blue')
plt.scatter(points[y_hc ==3,0], points[y_hc == 3,1], s=10, c='cyan')
plt.show()

# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(points, method='ward'))