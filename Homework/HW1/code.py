# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
file = open("dev.txt", "r")
#print (file.read())
dados = file.read()
data = dados.split()


j = 0
k = 0

data_organized = []
data_organized_t1 = []
data_organized_t2 = []
data_variance = []
linha1 = []
linha2 = []
data_mean1 = []
data_covariance1 = []
data_mean2 = []
data_covariance2 = []

while (data != []):
    data_organized.append(data[:27])
    data = data[27:]
variance = []
variance1 = []
variance2 = []
data_variance1 = []
data_variance2 = []
#variance = data_organized[0]
#print (np.var(variance))
for j in range (27):
    for i in range (len(data_organized)):
        if(data_organized[i][0] == '0'):
            variance1.append(data_organized[i][j])
        else:
            variance2.append(data_organized[i][j])
    variance1 = [float(i) for i in variance1]
    variance2 = [float(i) for i in variance2]
    data_variance1.append(np.var(variance1))
    data_variance2.append(np.var(variance2))
    data_mean1.append(np.mean(variance1))
    data_covariance1.append(np.cov(variance1))
    data_mean2.append(np.mean(variance2))
    data_covariance2.append(np.cov(variance2))
    del variance1[:]
    del variance2[:]
    
for j in range (27):
    for i in range(len(data_organized)):
        if(data_organized[i][0] == '0'):
            linha1.append(data_organized[i][j])
        else:
            linha2.append(data_organized[i][j])
    linha1 = [float(i) for i in linha1]
    linha2 = [float(i) for i in linha2]
    data_organized_t1.append(linha1)
    data_organized_t2.append(linha2)
    del linha1[:]
    del linha2[:]

#Data variance
line1 = []
line2 = []

#for j in range (26):
#    line1.append(data_organized_t1[j])
#    line2.append(data_organized_t2[j])
#    print(line1)
#    line1 = [float(i) for i in line1]
#    line2 = [float(i) for i in line2]
#    data_variance1.append(np.var(line1))
#    data_variance2.append(np.var(line2))
#    del line1[:]
#    del line2[:]
#    
    

#Histograms for the two datas with biggest variance  
histogram1 = []
histogram1 = data_organized_t1[9]
histogram1 = [float(i) for i in histogram1]
histogram1.sort()
histogram2 = []
histogram2 = data_organized_t1[25]
histogram2 = [float(i) for i in histogram2]
#plt.plot(histogram1, histogram2, 'ro', markersize=0.5)
plt.hist(histogram1, 50, density=1, facecolor='b')

histogram1 = []
histogram1 = data_organized_t2[9]
histogram1 = [float(i) for i in histogram1]
histogram1.sort()
histogram2 = []
histogram2 = data_organized_t2[25]
histogram2 = [float(i) for i in histogram2]
plt.hist(histogram1, 50, density=1, facecolor='r')
#plt.plot(histogram1, histogram2, 'bo', markersize=0.5)
plt.show()

##Gaussian Distribution
##Class 1
##parameters to set
#mu_x = data_mean2[26]
#variance_x = data_variance2[26]
#
#mu_y = data_mean2[25]
#variance_y = data_variance2[25]
#
##Create grid and multivariate normal
#x = []
#y = []
#for i in range (len(data_organized)):
#    x.append(data_organized[i][1])
#    y.append(data_organized[i][9])
#x = [float(i) for i in x]
#y = [float(i) for i in y]
#x.sort()
#y.sort()
##x = np.linspace(-10,10,500)
##y = np.linspace(-10,10,500)
#X, Y = np.meshgrid(x,y)
#pos = np.empty(X.shape + (2,))
#pos[:, :, 0] = X; pos[:, :, 1] = Y
#rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])
#
##Make a 3D plot
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.plot_surface(X, Y, rv.pdf(pos),cmap='hot',linewidth=0)
#ax.set_xlabel('X axis')
#ax.set_ylabel('Y axis')
#ax.set_zlabel('Z axis')
#plt.show()

#Greatest Overleap
k = 0
x = []
y = []
#
#for i in range(len(data_organized_t1)):
#    data_organized_t1[i] = [float(j) for j in data_organized_t1[i][j]]

#for i in range (26):
#    k += 1
#    for j in range(k,26):
#        histogram1 = []
#        histogram1 = data_organized_t1[i]
#        histogram1 = [float(i) for i in histogram1]
#        histogram1.sort()
#        histogram2 = []
#        histogram2 = data_organized_t1[j]
#        histogram2 = [float(i) for i in histogram2]
#        plt.plot(histogram1, histogram2, 'ro', markersize=0.5)
#        histogram1 = []
#        histogram1 = data_organized_t2[i]
#        histogram1 = [float(i) for i in histogram1]
#        histogram1.sort()
#        histogram2 = []
#        histogram2 = data_organized_t2[j]
#        histogram2 = [float(i) for i in histogram2]
#        plt.plot(histogram1, histogram2, 'bo', markersize=0.5)
#        plt.show()














