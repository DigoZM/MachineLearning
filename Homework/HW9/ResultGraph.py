# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:49:27 2020

@author: digo
"""
import numpy as np
import matplotlib.pyplot as plt

#Plot results
f_result = open('results.txt', 'r')
data = f_result.read()
f_result.close()
result = data.split()
result = map(float, result)
result_m = np.array(result)
x = len(result_m)/2
y = 2
result_m = result_m.reshape(x,y)
result_t = result_m.transpose()
new_scale = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
num_plots = len(result_m)/11
overlap = -10
for i in range(num_plots):
    start = i*11
    end = i*11 + 11
    actual_label = '{}'.format(overlap/float(10))
    overlap += 2
    plt.plot(new_scale, result_t[1][start:end], label = actual_label)
plt.legend(fontsize = 'small', loc = 1)
plt.title('Error Rate For Each Overlap')
plt.xlabel('K Neighbors in powers of 2')
plt.ylabel('Error Rate')
plt.xlim(-0.5, 12)
plt.show()