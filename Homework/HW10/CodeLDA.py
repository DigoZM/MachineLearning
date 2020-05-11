# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 19:03:20 2020

@author: digo
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal

#Reading File


##Train data
inicial_file = 'training_set.txt'
#inicial_file = 'test_set.txt'
f_train = open(inicial_file, 'r')
data = f_train.read()
f_train.close()
data_train = data.split()
data_train = map(float, data_train)
data_train_m = np.array(data_train)
x = len(data_train_m)/3
y = 3
data_train_m = data_train_m.reshape(x,y)
data_train_t = data_train_m.transpose()
half_t = len(data_train_m)/2
data_train_t0 = data_train_t[:,:half_t]
data_train_t1 = data_train_t[:,half_t:]
##Evaluation data
file_open = 'evaluation_set.txt'
#file_open = 'training_set.txt'
#file_open = 'test_set.txt'
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

#Ploting histogram of the two dimensions of original data set
#colors = ['b', 'r']
#for i in range (1, 3):
#    plt.hist([data_train_t0[i], data_train_t1[i]], color = colors, bins = 20)
#    graph_title = 'Histogram of feature {}'.format(i)
#    plt.title(graph_title)
#    plt.show()
    
#Find oprimal W
mean0 = np.array([np.mean(data_train_t0[1]), np.mean(data_train_t0[2])])
mean1 = np.array([np.mean(data_train_t1[1]), np.mean(data_train_t1[2])])

#S_B - between class cov matix
mean_all = (mean0 + mean1)/3
b_mean0 = mean0 - mean_all
b_mean1 = mean1 - mean_all
S_B_0 = np.dot(b_mean0[np.newaxis].T, b_mean0[np.newaxis])
S_B_1 = np.dot(b_mean1[np.newaxis].T, b_mean1[np.newaxis])
S_B_all = S_B_0 + S_B_1

#S_W total within class cov matrix
sc_mat_0 = np.zeros((2,2))
sc_mat_1 = np.zeros((2,2))
for i in range (half_t):
    #class 0
    row_0 = np.array([data_train_t0[1][i], data_train_t0[2][i]])
    row_0.reshape(2,1)
    mean0.reshape(2,1)
    var0 = (row_0 - mean0)
    sc_mat_0 += np.dot(var0[np.newaxis].T, var0[np.newaxis])
    #class 1
    row_1 = np.array([data_train_t1[1][i], data_train_t1[2][i]])
    row_1.reshape(2,1)
    mean1.reshape(2,1)
    var1 = (row_1 - mean0)
    sc_mat_1 += np.dot(var1[np.newaxis].T, var1[np.newaxis])
S_W_all = sc_mat_0 + sc_mat_1
eig_vals_all, eig_vec_all = np.linalg.eig(np.dot(np.linalg.inv(S_W_all), S_B_all))
eig_pairs_all = [(np.abs(eig_vals_all[i]), eig_vec_all[:,i]) for i in range (len(eig_vals_all))]

#sorting the pairs by eigenvalue , the higher, the more meaninfull is the vector
eig_pairs_all = sorted(eig_pairs_all, key=lambda k: k[0], reverse = True)
#print('Eigen pairs sorted by eigenvalue: \n')
#for i in eig_pairs_all:
#    print(i)
    
#Explained Variance
#print('Explained Variance:')
#ex_var = eig_vals_all[0]/sum(eig_vals_all)
#print(ex_var)
    
#Reducing dimension to one
optimal_w_all = np.hstack(eig_pairs_all[0][1].reshape(2,1))
#print('Optimal W:')
#print(optimal_w_all.real)

#Changing the data
Y_0_opt = np.dot(np.transpose(data_train_t0[1:,:]), optimal_w_all)
Y_1_opt = np.dot(np.transpose(data_train_t1[1:,:]), optimal_w_all)



##Plotting graph
#plt.hist([Y_0_opt, Y_1_opt], color = colors, bins = 20)
#plt.title('Histogram of Transformed data')
#plt.show()

#classifying data using MLE for one variable
##calculating patters
opt_mean0 = np.mean(Y_0_opt)
opt_mean1 = np.mean(Y_1_opt)
opt_var0 = np.var(Y_0_opt)
opt_var1 = np.var(Y_1_opt)

#Changing evaluate data
Y_ev_opt = np.dot(np.transpose(data_ev_t[1:,:]), optimal_w_all)
#plt.hist(Y_ev_opt, color = 'g', bins = 20)
#plt.title('Histogram of Transformed evaluation data')
#plt.show()

#Classifying
data_classified = []    
for i in range (len(data_ev_m)):
    p_class0 = norm.pdf(Y_ev_opt[i], opt_mean0, opt_var0)
    p_class1 = norm.pdf(Y_ev_opt[i], opt_mean1, opt_var1)
    if (p_class0 > p_class1):
        data_classified.append(0)
    else:
        data_classified.append(1)

#calculating error
error = 0
for i in range (len(data_classified)):
    if(data_classified[i] != data_ev_t[0][i]):
        error += 1
print('Error LDA')
print(error*100.0/len(data_classified))

############################
############################
#Now using PCA
pca = PCA(n_components=2, whiten=True)
X_all = data_train_t[1:,:].T
X0 = data_train_t0[1:,:].T
pca.fit(X_all)
X0_pca = pca.transform(X0)
X1 = data_train_t1[1:,:].T
X1_pca = pca.transform(X1)

#plt.scatter(X0_pca[:,0], X0_pca[:,1], color = 'b', alpha=0.2)
#plt.scatter(X1_pca[:,0], X1_pca[:,1], color = 'r', alpha=0.2)
#plt.show()
##Classify data with PCA
#Patterns
mean0_pca = [np.mean(X0_pca[:,0]),np.mean(X0_pca[:,1])]
mean1_pca = [np.mean(X1_pca[:,0]),np.mean(X1_pca[:,1])]
cov0_pca = np.cov(X0_pca.T)
cov1_pca = np.cov(X1_pca.T)
#Transforming Evaluation set
XE = data_ev_t[1:,:].T
XE_pca = pca.transform(XE)
#MLE
rv0 = multivariate_normal(mean0_pca, cov0_pca)
rv1 = multivariate_normal(mean1_pca, cov1_pca)
data_classified_mle = []
for i in range(len(XE_pca)):
    x = XE_pca[i]
    if (rv0.pdf(x) > rv1.pdf(x)):
        data_classified_mle.append(0)
    else:
        data_classified_mle.append(1)
#calculating error
error = 0
for i in range (len(data_classified_mle)):
    if(data_classified_mle[i] != data_ev_t[0][i]):
        error += 1
print('Error PCA with 2 dimensions:')
print(error*100.0/len(data_classified_mle))

#############################
#############################
##Now using PCA
#pca = PCA(n_components=1, whiten=True)
#X_all = data_train_t[1:,:].T
#X0 = data_train_t0[1:,:].T
#pca.fit(X_all)
#X0_pca = pca.transform(X0)
#X1 = data_train_t1[1:,:].T
#X1_pca = pca.transform(X1)
#
#
###Classify data with PCA
##Patterns
##mean0_pca = [np.mean(X0_pca[:,0]),np.mean(X0_pca[:,1])]
##mean1_pca = [np.mean(X1_pca[:,0]),np.mean(X1_pca[:,1])]
#mean0_pca = np.mean(X0_pca)
#mean1_pca = np.mean(X1_pca)
#cov0_pca = np.cov(X0_pca.T)
#cov1_pca = np.cov(X1_pca.T)
#plt.hist(X0_pca, color = 'b')
#plt.hist(X1_pca, color = 'r')
#plt.show()
##Transforming Evaluation set
#XE = data_ev_t[1:,:].T
#XE_pca = pca.transform(XE)
##MLE
#rv0 = multivariate_normal(mean0_pca, cov0_pca)
#rv1 = multivariate_normal(mean1_pca, cov1_pca)
#data_classified_mle = []
#for i in range(len(XE_pca)):
#    x = XE_pca[i]
#    if (rv0.pdf(x) > rv1.pdf(x)):
#        data_classified_mle.append(0)
#    else:
#        data_classified_mle.append(1)
##calculating error
#error = 0
#for i in range (len(data_classified_mle)):
#    if(data_classified_mle[i] != data_ev_t[0][i]):
#        error += 1
#print('Error PCA with 1 dimension:')
#print(error*100.0/len(data_classified_mle))

#Common MLE
cov0 = np.cov(data_train_t0[1:,:])
cov1 = np.cov(data_train_t1[1:,:])
#MLE
rv0 = multivariate_normal(mean0, cov0)
rv1 = multivariate_normal(mean1, cov1)
data_classified_mle = []
for i in range(len(data_ev_m)):
    x = [data_ev_m[i][1], data_ev_m[i][2]]
    if (rv0.pdf(x) > rv1.pdf(x)):
        data_classified_mle.append(0)
    else:
        data_classified_mle.append(1)
#calculating error
error = 0
for i in range (len(data_classified_mle)):
    if(data_classified_mle[i] != data_ev_t[0][i]):
        error += 1
print('Error simple MLE:')
print(error*100.0/len(data_classified_mle))


        



