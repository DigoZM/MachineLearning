# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:58:20 2020

@author: digo
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def writeResults(results, data):
    f_name = 'medeiros_rodrigo_5d_rf_{}.hyp'.format(data)
    file_object  = open(f_name, 'w')
    for clas in results:
        file_object.write('%d\n' % (clas))


def evaluate(model, test_features, test_labels, data):
    predictions = model.predict(test_features)
    writeResults(predictions, data)
    errors = 0
    for i in range (len(predictions)):
        if (predictions[i] != test_labels[i]):
          errors += 1
    print('Model Performance')
    print('Average Error: {:0.2f}%.'.format(errors*100.0/len(predictions)))
#End evaluation function    



if __name__ == "__main__":
    #Reading File
    ##Train data
    # f_train = open('train2D.txt', 'r')
    f_train = open('train5D.txt', 'r')
    data_train = np.genfromtxt(f_train)
    f_train.close()
    #f_test0 = open('/content/drive/My Drive/Machine Learning/Final Exam/train2D.txt', 'r')
    # f_test0 = open('train2D.txt', 'r')
    f_test0 = open('train5D.txt', 'r')
    data_test0 = np.genfromtxt(f_test0)
    f_test0.close()
    #f_test1 = open('/content/drive/My Drive/Machine Learning/Final Exam/dev2D.txt', 'r')
    # f_test1 = open('dev2D.txt', 'r')
    f_test1 = open('dev5D.txt', 'r')
    data_test1 = np.genfromtxt(f_test1)
    f_test1.close()
    #f_test2 = open('/content/drive/My Drive/Machine Learning/Final Exam/eval2D.txt', 'r')
    # f_test2 = open('eval2D.txt', 'r')
    f_test2 = open('eval5D.txt', 'r')
    data_test2 = np.genfromtxt(f_test2)
    f_test2.close()
    #print('Data read')

    ##Arranging Data
    #Traning
    train_features = data_train[:,1:]
    train_labels = data_train[:,0]
    #Testing
    test_features0 = data_test0[:,1:]
    test_labels0 = data_test0[:,0]
    test_features1 = data_test1[:,1:]
    test_labels1 = data_test1[:,0]
    test_features2 = data_test2[:,1:]
    test_labels2 = data_test2[:,0]
    
    # Instantiate model with best parameters found
    rf = RandomForestClassifier(n_estimators = 2400, min_samples_split = 15, min_samples_leaf = 2, max_features = 'auto', max_depth = 10, bootstrap = False)
    #print('tree created')
    
    # Train the model on training data
    rf.fit(train_features, train_labels);
    #print('data trained')
    
    # Use the forest's predict method on the test data train.txt
    print('Test on train.txt:')
    evaluate(rf, test_features0, test_labels0, 'train')
    
    # Use the forest's predict method on the test data dev.txt
    print('Test on dev.txt:')
    random_accuracy = evaluate(rf, test_features1, test_labels1, 'dev')
    
    # Use the forest's predict method on the test data eval.txt
    print('Test on eval.txt:')
    random_accuracy = evaluate(rf, test_features2, test_labels2, 'eval')
