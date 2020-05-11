# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 00:07:54 2020

@author: digo
"""
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

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
file_open = 'training_set.txt'
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

#Train Classifyer -- Calculate patterns
data_mean0 = []
data_mean1 = []
for i in range (1,3):
    data_mean0.append(np.mean(data_train_t0[i]))
    data_mean1.append(np.mean(data_train_t1[i]))
data_cov0 = np.cov(data_train_t0[0:][1:])
data_cov1 = np.cov(data_train_t1[0:][1:])

#Classify data
#calculating likelihood THE RIGHT WAY!!!
data_classified = []
rv0 = multivariate_normal(data_mean0, data_cov0)
rv1 = multivariate_normal(data_mean1, data_cov1)
for i in range(len(data_ev_m)):
    x = data_ev_m[i][1:]
    if (rv0.pdf(x) > rv1.pdf(x)):
        data_classified.append(0)
    else:
        data_classified.append(1)
#Calculating error
erro = 0
for j in range(len(data_ev_m)):
    if (data_classified[j] != data_ev_m[j][0]):
        erro += 1
error_rate = erro/float(len(data_ev_m))

#Drawing decision surface
X = data_ev_t[1:]
X = X.transpose()
Y = data_ev_t[0]
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

## get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]
plt.plot(xx, yy, 'k-')

#Plotting
plt.scatter(data_ev_t0[1], data_ev_t0[2], color = 'b')
plt.scatter(data_ev_t1[1], data_ev_t1[2], color = 'r')
plt.title('evaluation.txt')
plt.xlim(-0.5, 1)
plt.ylim(-0.5, 1)
