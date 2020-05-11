# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 12:51:38 2020

@author: digo
"""     

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np


#Reading File
##Train data
f_train = open('training_set.txt', 'r')
data_train_m = np.genfromtxt(f_train)
data_train_t = data_train_m.transpose()
f_train.close()
##Evaluation data
file_open = 'evaluation_set.txt'
#file_open = 'training_set.txt'
f_train = open(file_open, 'r')
data_ev_m = np.genfromtxt(f_train)
data_ev_t = data_ev_m.transpose()
f_train.close()
#Arranging Data
X = data_train_m[:,1:]
y = data_train_m[:,0]


#RandomForestClassifier
#X, y = make_classification(n_samples=1000, n_features=4,
#                           n_informative=2, n_redundant=0,
#                           random_state=0, shuffle=False)
clf = RandomForestClassifier(random_state=0, n_estimators = 100)
clf.fit(X, y)

#print(clf.feature_importances_)
prediction = []
for i in data_ev_m:
#    print(i[1:])
    prediction.append(clf.predict([i[1:]]))
    
#Calculating Error Rate
error = 0
for i in range(len(prediction)):
    if(prediction[i] != data_ev_t[0][i]):
        error += 1
error_rate = float(error)/len(data_ev_m)
print('Decision Tree Error:')
print(error_rate*100.00)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    