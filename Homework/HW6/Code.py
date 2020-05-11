# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 21:24:30 2020

@author: digo
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.patches import Ellipse
import numpy as np

#data
w1 = [[0, 2, 2, 0],[0, 0, 2, 2]]
w2 = [[1, 2, 1, 3], [1, 1, 2, 3]]


#Train Classifyer -- Calculate patterns
data_mean0 = []
data_mean1 = []
for i in range (2):
    data_mean0.append(np.mean(w1[i]))
    data_mean1.append(np.mean(w2[i]))
data_cov0 = np.cov(w1)
data_cov1 = np.cov(w2)


#fig = plt.figure()
#r = 2 #Radius
#a = 2*r #width 
#b = r #height
##center of the ellipse
#x = 2
#y = 2
#ax = fig.subplots(1,1)
#
#ax.add_patch(Arc((x, y), a, b, angle = 45,
#             theta1=0, theta2=360, 
#             edgecolor='black', lw=1.1))
#r = 0.5 #Radius
#a = r #width 
#b = r #height
#
##center of the ellipse
##x = 2
##y = 2
##ax = fig.plt(1,1)
##
##ax.add_patch(Arc((x, y), a, b, angle = 45,
##             theta1=0, theta2=360, 
##             edgecolor='black', lw=1.1))
#ellipse = Ellipse(xy=(2., 2), width=r, height=r, angle=0, fill = False)
#ax.add_patch(ellipse)

#Now look for the ends of the Arc and manually set the limits
#plt.plot([x,0.687],[y,0.567], color='r',lw=1.1)
#plt.plot([x,0.248],[y,0.711], color='r',lw=1.1)


#ploting scatter 


plt.scatter(w1[0], w1[1], color='r', label = 'w1')
plt.scatter(w2[0], w2[1], color = 'b', label = 'w2')
plt.plot([0,2.5], [2.5,0], color = 'black')
plt.legend()
plt.show()

##Plot B
#Priors = [0, 0.5, 1]
#Error = [0.5, 0.25, 0.5]
#plt.plot(Priors, Error, color = 'g')
#plt.xlabel('Alpha')
#plt.ylabel('P(E)')
#plt.title('Relation between prior and probability of error')
#plt.show()