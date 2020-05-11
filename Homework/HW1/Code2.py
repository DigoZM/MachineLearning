# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 20:32:51 2020

@author: digo
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

#Read the Data in the file
file = open("dev.txt", "r")
dados = file.read()
data = dados.split()
data_aux = []
data_o = []
data_t0 = []
data_t1 = []
aux = []
aux0 = []
aux1 = []

while(data != []):
    for i in range(27):
        aux.append(float(data[i]))        
    data_o.append(aux)
    data = data[27:]
    aux = aux[27:]

for i in range (27):    
    for j in range (len(data_o)):
        if(data_o[j][0] == 0):
            aux0.append(data_o[j][i])
        else:
            aux1.append(data_o[j][i])
    data_t0.append(aux0)
    data_t1.append(aux1)
    aux0 = aux0[len(aux0):]
    aux1 = aux1[len(aux1):]
###############
#Look for data with greatest variance
data_var0 = []
data_var1 = []

for i in range (27):
    data_var0.append(np.var(data_t0[i]))
    data_var1.append(np.var(data_t1[i]))

#Do the scatter plot with these datas
x = 1
y = 9
#plt.plot(data_t0[x], data_t0[y], 'ro', markersize=1, label = 'Class 0')
#plt.plot(data_t1[x], data_t1[y], 'bo', markersize=1, label = 'Class 1')    
#plt.xlabel("Feature 1")
#plt.ylabel("Feature 9")
#plt.title("Feature with 2 greatest variance")
#plt.legend()
#plt.show()     

#Estimating mean and covariance of each class
data_mean0 = []
data_mean1 = []
data_mean0.append(np.mean(data_t0[x]))
data_mean0.append(np.mean(data_t0[y]))
data_mean1.append(np.mean(data_t1[x]))
data_mean1.append(np.mean(data_t1[y]))
#data_cov0 = np.cov(data_t0[x], data_t0[y])
#data_cov1 = np.cov(data_t1[x], data_t1[y])
#
##Plotting support regions
#fig = plt.figure()
#ax1= fig.add_subplot(1, 1, 1)
#ax1.scatter(data_t0[x], data_t0[y], c=("blue"), label = 'Class0')
#ax1.scatter(data_t1[x], data_t1[y], c=("green"), label = 'Class1')
#
#w0,v0 = np.linalg.eigh(data_cov0)
#w1,v1 = np.linalg.eigh(data_cov1)
#angle0 = np.degrees(np.arctan(v0[1,1] / v0[0,1]))
#angle1 = np.degrees(np.arctan(v1[1,1] / v1[0,1]))
#gaus_ellipse0 = patches.Ellipse((data_mean0[0], data_mean0[1]),4*np.sqrt(w0[1]),6*np.sqrt(w0[0]), angle=angle0,alpha=0.3,facecolor='blue')            
#gaus_ellipse1 = patches.Ellipse((data_mean1[0], data_mean1[1]),4*np.sqrt(w1[1]),6*np.sqrt(w1[0]), angle=angle1,alpha=0.1,facecolor='green')
#ax1.add_patch(gaus_ellipse0)
#ax1.add_patch(gaus_ellipse1)
##ax1.xlabel("Feature 1")
##ax1.ylabel("Feature 9")
##ax1.title("Feature with 2 greatest variance")
#ax1.legend()
#plt.show()

#Looking for the feature with largest covariance
#x=1
#y=16
#data_cov_big = np.cov(data_t0[x], data_t1[x])
#data_cov_test = np.cov(data_t0[y], data_t1[y])
#plt.plot(data_t0[x], data_t0[y], 'ro', markersize=1, label = 'Class 0')
#plt.plot(data_t1[x], data_t1[y], 'bo', markersize=1, label = 'Class 1')    
#plt.xlabel("Feature 1")
#plt.ylabel("Feature 16")
#plt.title("Two features")
#plt.legend()
#plt.show()   
#colors = ['blue', 'green']
#labels = ['Class 0', 'Class 1']
#plt.hist([data_t0[x], data_t1[x]], bins = 20, normed = True, color = colors, label = labels)
#plt.legend()
#plt.show()

#Gaussian Distribution
#Class 1
#parameters to set
x = 1
y = 9
mu_x = data_mean1[0]


mu_y = data_mean1[1]
data_cov0 = np.cov(data_t1[x], data_t1[y])

#Create grid and multivariate normal
xv = []
yv = []
xv = data_t1[x]
xv.sort()
yv = data_t1[y]
yv.sort()
#x = np.linspace(-10,10,500)
#y = np.linspace(-10,10,500)
X, Y = np.meshgrid(xv,yv)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y
rv = multivariate_normal([mu_x, mu_y], data_cov1)

#Make a 3D plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 9')
ax.set_zlabel('Density')
plt.show()
    
