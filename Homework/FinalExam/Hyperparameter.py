# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 16:21:08 2020

@author: digo
"""

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = 0
    for i in range (len(predictions)):
        if (predictions[i] != test_labels[i]):
          errors += 1
    print('Total Erros:')
    print(errors)
    print(len(predictions))
    print(len(test_labels))
#    mape = 100 * np.mean(errors / test_labels)
#    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.2f}%.'.format(errors*100.0/len(predictions)))
#    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return predictions

#Reading File
##Train data
#f_train = open('/content/drive/My Drive/Machine Learning/Final Exam/train.txt', 'r')
f_train = open('train2D.txt', 'r')
data_train = np.genfromtxt(f_train)
f_train.close()
#f_test0 = open('/content/drive/My Drive/Machine Learning/Final Exam/train2D.txt', 'r')
f_test0 = open('train2D.txt', 'r')
data_test0 = np.genfromtxt(f_test0)
f_test0.close()
#f_test1 = open('/content/drive/My Drive/Machine Learning/Final Exam/dev2D.txt', 'r')
f_test1 = open('dev2D.txt', 'r')
data_test1 = np.genfromtxt(f_test1)
f_test1.close()
#f_test2 = open('/content/drive/My Drive/Machine Learning/Final Exam/eval2D.txt', 'r')
f_test2 = open('eval2D.txt', 'r')
data_test2 = np.genfromtxt(f_test2)
f_test2.close()

print('Data read')

##Arranging Training Data
train_features = data_train[:,1:]
train_labels = data_train[:,0]
print('Data arranged')
##Arranging Test Data
#A = data_train[:30000,1:]
#B = data_train[70000:,1:]
#test_features = np.concatenate([A, B])
#A = data_train[:30000,0]
#B = data_train[70000:,0]
#test_labels = np.concatenate([A, B])
#test_features = train_features
#test_labels = train_labels
test_features0 = data_test0[:,1:]
test_labels0 = data_test0[:,0]
test_features1 = data_test1[:,1:]
test_labels1 = data_test1[:,0]
test_features2 = data_test2[:,1:]
test_labels2 = data_test2[:,0]

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 600, stop = 2400, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [ 5, 10, 15]

# Minimum number of samples required at each leaf node
min_samples_leaf = [ 2, 4, 6]

# Method of selecting samples for training each tree
bootstrap = [False]# Create the random grid

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
               
#print(random_grid)
print('best')               
base_model = RandomForestClassifier(n_estimators = 2400, min_samples_split = 15, min_samples_leaf = 2, max_features = 'auto', max_depth = 10, bootstrap = False)
base_model.fit(train_features, train_labels)
print('Data trained')
base_accuracy = evaluate(base_model, test_features0, test_labels0)
base_accuracy = evaluate(base_model, test_features1, test_labels1)
base_accuracy = evaluate(base_model, test_features2, test_labels2)
         
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 200, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(train_features, train_labels)
print('Data trained')
best_random = rf_random.best_estimator_
print('Test on train.txt:')
random_accuracy = evaluate(best_random, test_features0, test_labels0)
print('Test on dev.txt:')
random_accuracy = evaluate(best_random, test_features1, test_labels1)
print('Test on eval.txt:')
random_accuracy = evaluate(best_random, test_features2, test_labels2)
print(rf_random.best_params_)
#Average Error: 8.90%.
#{'max_depth': 10, 'n_estimators': 2200, 'min_samples_split': 10, 'bootstrap': False, 'min_samples_leaf': 1, 'max_features': 'sqrt'}

#5.44, 8.25, 8.60
#{'n_estimators': 2400, 'min_samples_split': 15, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 10, 'bootstrap': False}