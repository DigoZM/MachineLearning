# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:08:03 2020

@author: digo
"""
import matplotlib.pyplot as plt
import numpy as np
#Parameters
mean0 = [0, 0]
mean1 = [1, 0]
cov0 = [[2, 0], [0, 1]]
cov1 = [[1, 0], [0, 2]]
points = 10000

#Generating points
x0, y0 = np.random.multivariate_normal(mean0, cov0, points).T
x1, y1 = np.random.multivariate_normal(mean1, cov1, points).T

#write data points in a file
#file_object  = open('evaluation_set.txt', 'w')
for i in range (len(x0)):
    file_object.write('%d %.17f %.17f\n' % (0, x0[i], y0[i]))
for i in range (len(x0)):
    file_object.write('%d %.17f %.17f\n' % (1, x1[i], y1[i]))
file_object.close()

#Plot
plt.scatter(x0, y0, color = 'b')
plt.scatter(x1, y1, color = 'r')
plt.title("Evaluation Set")
plt.axis('equal')
plt.show()
