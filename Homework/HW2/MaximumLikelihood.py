# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 21:25:20 2020

@author: digo
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from numpy import *
import math

def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0/ ( math.pow((2*pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = matrix(x - mu)
        inv = sigma.I        
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")



#Read the Data in the file
file = open("train.txt", "r")
dados = file.read()
data = dados.split()
data_aux = []
data_o = []
data_t0 = []
data_t1 = []
aux = []
aux0 = []
aux1 = []
test = []


while(data != []):
    for i in range(27):
        aux.append(float(data[i]))
        test.append(float(data[i]))        
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
    
#test2 = np.array(test)
#shape = ( len(data_o), 27 )
#test2.reshape( shape )
###############

#Calculating the patterns
data_mean0 = []
data_mean1 = []
data_sd0 = []
data_sd1 = []
data_var0 = []
data_var1 = []
data_cov0 = []
data_cov1 = []


for i in range (27):
    data_mean0.append(np.mean(data_t0[i]))
    data_mean1.append(np.mean(data_t1[i]))
    data_sd0.append(np.std(data_t0[i]))
    data_sd1.append(np.std(data_t1[i]))
    data_var0.append(np.var(data_t0[i]))
    data_var1.append(np.var(data_t1[i]))
    
for i in range (26):
    for j in range (26):
        if(j == i):
            aux0.append(data_var0[i])
            aux1.append(data_var1[i])
        else:
            aux0.append(0)
            aux1.append(0)
    data_cov0.append(aux0)
    data_cov1.append(aux1)
    aux0 = aux0[len(aux0):]
    aux1 = aux1[len(aux1):]
          
data_cov_all0 = np.cov(data_t0[0:][1:])
data_cov_all1 = np.cov(data_t1[0:][1:])
    
print('Patterns Calculated\n')
    
#Opening the train file
file = open("dev.txt", "r")
train_dados = file.read()
train_data = train_dados.split()
train_data_o = []
train_data_t = []

while(train_data != []):
    for i in range(27):
        aux.append(float(train_data[i]))        
    train_data_o.append(aux)
    train_data = train_data[27:]
    aux = aux[27:]

train_data_t0 = []
train_data_t1 = []

for i in range (27):    
    for j in range (len(train_data_o)):
        if(train_data_o[j][0] == 0):
            aux0.append(train_data_o[j][i])
        else:
            aux1.append(train_data_o[j][i])
        aux.append(train_data_o[j][i])
    train_data_t0.append(aux0)
    train_data_t1.append(aux1)
    train_data_t.append(aux)
    aux = aux[len(aux):]
    aux0 = aux0[len(aux0):]
    aux1 = aux1[len(aux1):]

print('File read\n')


##Ploting the probability density functions
#
##for i in range (2)
##    x = train_data_t0[i]
##    x.sort()
##    plt.plot(x, norm.pdf(x))
##plt.xlim(-10,10)
#x = train_data_t0[3]
#y = train_data_t1[3]
#x.sort()
#y.sort()
#plt.plot(x, norm.pdf(x))
#plt.plot(y, norm.pdf(y))
#plt.show


#Calculating the likelihood
data_c = []
    
data_c.append(train_data_t[0])

j = 1

for j in range (1,27):
    for i in range(len(train_data_o)):
        p_class0 = norm.pdf(train_data_t[j][i], data_mean0[j], data_sd0[j])
        p_class1 = norm.pdf(train_data_t[j][i], data_mean1[j], data_sd1[j])
        if(p_class0 > p_class1):
            aux.append(0)
        else:
            aux.append(1)
    data_c.append(aux)
    aux = aux[len(aux):]
    
#Calculating error
erro_c = []
for i in range (1,27):
    erro = 0
    for j in range(len(train_data_o)):
        if (data_c[0][j] != data_c[i][j]):
            erro += 1
    temp = erro/float(len(train_data_o))
    erro_c.append(temp)

#Plotting Error Bars
y_pos = np.arange(1, len(erro_c)+1)
plt.bar(y_pos, erro_c)
plt.xlabel('Features')
plt.ylabel('Error Rate')
plt.title('Classification of dev.txt')
plt.show()


##Calculating likelihood using two features
#data_c2 = []
#for i in range(len(train_data_o)):
#    j = 15
#    p1_class0 = norm.pdf(train_data_t[j][i], data_mean0[j], data_sd0[j])
#    p1_class1 = norm.pdf(train_data_t[j][i], data_mean1[j], data_sd1[j])
#    j = 9
#    p2_class0 = norm.pdf(train_data_t[j][i], data_mean0[j], data_sd0[j])
#    p2_class1 = norm.pdf(train_data_t[j][i], data_mean1[j], data_sd1[j])
#    if(p1_class0*p2_class0 > p_class1*p2_class0):
#        aux.append(0)
#    else:
#        aux.append(1)
#data_c2.append(aux)
#aux = aux[len(aux):]
##Calculating error from two features
#erro = 0
#for j in range(len(train_data_o)):
#    if (train_data_t[0][j] != data_c2[0][j]):
#        erro += 1
#temp = erro/float(len(train_data_o))


##calculating likelihood THE RIGHT WAY!!!
#data_c = []
#for i in range(len(train_data_o)):
#    x = train_data_o[i][1:]
#    rv0 = multivariate_normal(data_mean0[1:], data_cov_all0)
#    rv1 = multivariate_normal(data_mean1[1:], data_cov_all1)
#    if (rv0.pdf(x) > rv1.pdf(x)):
#        data_c.append(0)
#    else:
#        data_c.append(1)
#        
##Calculating error
#erro = 0
#for j in range(len(train_data_o)):
#    if (data_c[j] != train_data_t[0][j]):
#        erro += 1
#error_rate = erro/float(len(train_data_o))  

##Assuming priors are not equal from THE RIGHT WAY!!!
#data_c = []
#error_rate = []
#rv0 = multivariate_normal(data_mean0[1:], data_cov_all0)
#rv1 = multivariate_normal(data_mean1[1:], data_cov_all1)
#for prior in range (101):
#    print 'Calculating prior', prior
#    for i in range(len(train_data_o)):
#        x = train_data_o[i][1:]
#        p_class0 = (rv0.pdf(x))*(prior/float(100))
#        p_class1 = (rv1.pdf(x))*(1-(prior/float(100)))
#        if (p_class0 > p_class1):
#            data_c.append(0)
#        else:
#            data_c.append(1)
#    #Calculating error
#    erro = 0
#    for j in range(len(train_data_o)):
#        if (data_c[j] != train_data_t[0][j]):
#            erro += 1
#    error_rate.append(erro/float(len(train_data_o)))
#    data_c = []
#
#y = np.arange(0,1.01,0.01)
#plt.plot(y, error_rate)
#plt.xlabel('Prior')
#plt.ylabel('Error Rate')
#plt.title('dev.txt')
#plt.xlim(0,1)
#plt.show
#                

        
##Calculating likelihood using multivariate normal
#data_c = [] 
#for i in range (len(train_data_o)):
#    x = train_data_t[1:]
#    p_class0 = multivariate_normal.pdf(x, data_mean0[1:], data_cov_all0)
#    p_class1 = multivariate_normal.pdf(x, data_mean1[1:], data_cov_all1)
#    if(p_class0 > p_class1):
#        aux.append(0)
#    else:
#        aux.append(1)
#    data_c.append(aux)
#    aux = aux[len(aux):]
##Calculating error
#erro = 0
#for j in range(len(train_data_o)):
#    if (data_c[0][j] != data_c[i][j]):
#        erro += 1
#temp = erro/float(len(train_data_o))

##Assuming Priors are not equal
#data_c = []
#    
#
#
#j = 1
#error_all = []
#for prior in range (11):
#    data_c.append(train_data_t[0])
#    for j in range (1,27):
#        for i in range(len(train_data_o)):
#            p_class0 = norm.pdf(train_data_t[j][i], data_mean0[j], data_sd0[j])*(prior/float(10))
#            p_class1 = norm.pdf(train_data_t[j][i], data_mean1[j], data_sd1[j])*(1-(prior/float(10)))
#            if(p_class0 > p_class1):
#                aux.append(0)
#            else:
#                aux.append(1)
#        data_c.append(aux)
#        aux = aux[len(aux):]
#    print "prior", prior,"calculated\n"
#    #Calculating error
#    error_c = []
#    for i in range (27):
#        erro = 0
#        for j in range(len(train_data_o)):
#            if (data_c[0][j] != data_c[i][j]):
#                erro += 1
#        temp = erro/float(len(train_data_o))
#        error_c.append(temp)
#    error_all.append(error_c)
#    error_c = error_c[len(error_c):]
#    data_c = data_c[len(data_c):]
#
#error_t = []
#x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
#print('begin to plot error\n')
#
#for j in range (1, len(error_all[0])):
#    for i in range (len(error_all)):
#        aux.append(error_all[i][j])
#    error_t.append(aux)
#    aux = aux[len(aux):]
#for i in range (len(error_t)):
#    j = i + 1
#    labels = "Feature %d" % (j)
#    plt.plot(x,error_t[i], label = labels)
#    
#plt.legend(fontsize = 'xx-small', loc ='upper right')
#plt.title('train.txt data')
#plt.xlabel('Prior')
#plt.ylabel('Error Rate')
#plt.show()





















