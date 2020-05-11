# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 21:34:20 2020

@author: digo
"""

import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import math 
from sklearn.decomposition import PCA

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

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

#TRANSFORMING TRAIN DATA
aux = data_train_m[:][:].T
aux = aux[1:].T
pca0 = PCA(2)
pca0.fit(aux)
#print(pca0.components_)
#print(pca0.explained_variance_)
new0 = pca0.transform(aux)
#pca1 = PCA(2)
#pca1.fit(aux[10000:])
##print(pca1.components_)
##print(pca1.explained_variance_)
#new1 = pca1.transform(aux[10000:])
aux0 = new0.T
#aux1 = new1.T
#transforming evaluation
aux = data_ev_m[:][:].T
aux = aux[1:].T
pca = PCA(2)
pca.fit(aux)
#print(pca.components_)
#print(pca.explained_variance_)
new_ev = pca.transform(aux)
aux_ev = new_ev.T

#Plotting data transformed
plt.scatter(aux0[0][:10000], aux0[1][:10000], color = 'b')
plt.scatter(aux0[0][10000:], aux0[1][10000:], color = 'r')
#plt.scatter(aux_ev[0][:10000], aux_ev[1][:10000], color = 'b')
#plt.scatter(aux_ev[0][10000:], aux_ev[1][10000:], color = 'r')
data0 = aux0[:,:10000]
data1 = aux0[:,10000:]
print(np.mean(data0[0]), np.mean(data0[1]))
cov0 = np.cov(data0)
print(np.cov(aux0[:10000]))
print('outro')
print(np.mean(data1[0]), np.mean(data1[1]))
cov1 = np.cov(data1)
print(np.cov(aux0[10000:]))
#plt.scatter(data_train_t0[1], data_train_t0[2], color = 'b')
#plt.scatter(data_train_t1[1], data_train_t1[2], color = 'r')
plt.title('Training Data Transformed')



##Train Classifyer -- Calculate patterns
#data_mean0 = []
#data_mean1 = []
#for i in range (1,3):
#    data_mean0.append(np.mean(data_train_t0[i]))
#    data_mean1.append(np.mean(data_train_t1[i]))
#data_cov0 = np.cov(data_train_t0[0:][1:])
#data_cov1 = np.cov(data_train_t1[0:][1:])

#Train classifier for transformed data
data_mean0 = [np.mean(aux0[0][10000:]), np.mean(aux0[1][:10000])]
data_mean1 = [np.mean(aux0[0][:10000]), np.mean(aux0[1][10000:])]
data_cov0 = np.cov(aux.T)
data_cov1 = np.cov(aux.T)

##Classifier Using the mean

#classified_mean = []
#for i in range (20000):
#    distance0 = math.sqrt( ((data_mean0[0]-new_ev[i][0])**2)+((data_mean0[1]-new_ev[i][1])**2) )
#    distance1 = math.sqrt( ((data_mean1[0]-new_ev[i][0])**2)+((data_mean1[1]-new_ev[i][1])**2) )
#    if(distance0 > distance1):
#        classified_mean.append(0)
#    else:
#        classified_mean.append(1)
        
        
##Classify data
##calculating likelihood THE RIGHT WAY!!!
#classified_likelihood = []
#rv0 = multivariate_normal(data_mean0, data_cov0)
#rv1 = multivariate_normal(data_mean1, data_cov1)
#for i in range(len(data_ev_m)):
#    x = data_ev_m[i][1:]
#    if (rv0.pdf(x) < rv1.pdf(x)):
#        classified_likelihood.append(0)
#    else:
#        classified_likelihood.append(1)
##        
##Calculating error
#erro = 0
#for j in range(20000):
#    if (classified_likelihood[j] != data_ev_m[j][0]):
#        erro += 1
#error_rate = erro/float(len(data_ev_m))

##Principal Component Analyses
#class0 = data_train_t0[:][1:3].T
#pca0 = PCA(n_components=2)
#pca0.fit(class0)
#class1 = data_train_t1[:][1:3].T
#pca1 = PCA(n_components=2)
#pca1.fit(class1)
#    #Plotting
##plt.scatter(class0[:, 0], class0[:, 1], alpha=0.2, color = 'b')
##for length, vector in zip(pca0.explained_variance_, pca0.components_):
##    v = -vector * 3 * np.sqrt(length)
##    draw_vector(pca0.mean_, pca0.mean_ + v)
##plt.scatter(class1[:, 0], class1[:, 1], alpha=0.2, color = 'r')
##for length, vector in zip(pca1.explained_variance_, pca1.components_):
##    v = -vector * 3 * np.sqrt(length)
##    draw_vector(pca1.mean_, pca1.mean_ + v)
##plt.axis('equal');
#    #Transforming
#pca0 = PCA(n_components=1)
#pca0.fit(class0)
#class0_pca = pca0.transform(class0)
#class0_new = pca0.inverse_transform(class0_pca)
##plt.scatter(class0[:, 0], class0[:, 1], alpha=0.2, color ='b')
#pca1 = PCA(n_components=1)
#pca1.fit(class1)
#class1_pca = pca1.transform(class1)
#class1_new = pca1.inverse_transform(class1_pca)
##plt.scatter(class1[:, 0], class1[:, 1], alpha=0.2, color = 'r')
#plt.scatter(class0_new[:, 0], class0_new[:, 1], alpha=0.2, color = 'b')
#plt.scatter(class1_new[:, 0], class1_new[:, 1], alpha=0.2, color = 'r')
#plt.axis('equal');

##Classify data transformed
#data_mean0 = []
#data_mean1 = []
#for i in range (2):
#    data_mean0.append(np.mean((class0_new[:, i].T)))
#    data_mean1.append(np.mean((class1_new[:, i].T)))
#data_cov0 = np.cov((class0_new.T))
#data_cov1 = np.cov((class1_new.T))

##calculating likelihood THE RIGHT WAY!!!
#
#classified_likelihood = []
##rv0 = multivariate_normal(data_mean0, data_cov0)
#rv1 = multivariate_normal(data_mean1, data_cov1)
#for i in range(len(data_ev_m)):
#    x = data_ev_m[i][1:]
#    if (rv0.pdf(x) > rv1.pdf(x)):
#        classified_likelihood.append(0)
#    else:
#        classified_likelihood.append(1)
#        
##Calculating error
#erro = 0
#for j in range(len(data_ev_m)):
#    if (classified_likelihood[j] != data_ev_m[j][0]):
#        erro += 1
#error_rate = erro/float(len(data_ev_m))
#
#
##Drawing decision surface gaussian
#X = data_ev_t[1:]
#X = X.transpose()
#Y = data_ev_t[0]
#clf = svm.SVC(kernel='linear')
#clf.fit(X, Y)
#
### get the separating hyperplane
#w = clf.coef_[0]
#a = -w[0] / w[1]
#xx = np.linspace(-5, 5)
#yy = a * xx - (clf.intercept_[0]) / w[1]
#plt.plot(xx, yy, 'k-')

##Drawingdesicion surface mean
#x = [(data_mean1[0] + data_mean0[0])/2, (data_mean1[0] + data_mean0[0])/2]
#y = [-6, 6]
#plt.plot(x, y, color = 'black')
#
##Plotting
#plt.scatter(data_ev_t0[1], data_ev_t0[2], color = 'b')
#plt.scatter(data_ev_t1[1], data_ev_t1[2], color = 'r')
#plt.scatter(data_mean0[0], data_mean0[1], color = 'g')
#plt.scatter(data_mean1[0], data_mean1[1], color = 'white')
#plt.title('training.txt')
#plt.ylim(-6, 6)
#plt.xlim(-9,9)
plt.show()


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    