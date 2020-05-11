# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 12:21:48 2020

@author: digo
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

#Read file
data_sum_v = []
for i in range (1,11):
    file_name = 'set{}.txt'.format(i)
    #file_name = 'settest.txt'
    f_data = open(file_name, 'r')
    data = f_data.read()
    f_data.close()
    data_set = data.split()
    data_set = map(float, data_set)
    aux = []
    aux.append(data_set[0])
    for i in range (1, len(data_set)):  
        aux.append(aux[i-1] + data_set[i])
    data_sum_v.append(aux)

#Estimate the mean using a maximum likelihood estimate
real_mean = np.mean(data_set)
means = []
distance = []
n_array = []
sum_data = 0
#Creating n array
arr1 = np.arange(1000)
arr2 = np.arange(1000, 10000, 10)
arr3 = np.arange(10000,100000,100)
#arr4 = []
arr4 = np.arange(100000,1000000,1000)
n_array = np.append(np.append(arr1, arr2), np.append(arr3, arr4))
#n_array = np.arange(10)
n_array = map(int, n_array)
x_array = []

#Suming
data_sum_all = []
for i in range(len(data_set)):
    aux = 0
    for j in range (10):
        aux = aux + data_sum_v[j][i]
    data_sum_all.append(aux/10)        

#
#for i in range(len(n_array)-1):
#    #sampling = random.sample(data_set, n_array[i])
#    #mean_estimated = np.mean(sampling)
#    mean_estimated = data_sum_all[n_array[i]]/(n_array[i]+1)
#    means.append(mean_estimated)
#    distance.append((real_mean-mean_estimated)**2)
#    x_array.append(n_array[i])

#Estimating Mean using Bayesian estimate
results = []
for i in range (len(n_array) - 1):
    mean_estimated = data_sum_all[n_array[i]]/(n_array[i] + 1)
    n = n_array[i]
    sigma = 1.0
    sigma0 = 1.0
    mean_prior = 2.0
    term1 =float( ((n*(sigma**2))/((n*(sigma**2))+(sigma**2))))
    term2 = ((sigma**2)/((n*(sigma**2))+sigma**2))
    result = (term1*mean_estimated) + (term2*mean_prior)
    results.append(result)   
    means.append(mean_estimated)
    distance.append((real_mean-result)**2)
    x_array.append(n_array[i])


#Plotting Results
plt.plot(x_array, distance)
plt.xlabel('Number of Points')
plt.ylabel('Error')
plt.title('All Sets')
plt.show()
data_set = []