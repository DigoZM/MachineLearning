# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:17:24 2020

@author: digo
"""

# pytorch mlp for binary classification
import numpy as np
from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_

def writeResults(results, data):
    f_name = 'results{}'.format(data)
    file_object  = open(f_name, 'w')
    for clas in results:
        file_object.write('%d\n' % (clas))

# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        # df = read_csv(path, header=None)
        #my turn
        f_train = open(path, 'r')
        data_train = np.genfromtxt(f_train)
        f_train.close()
        # store the inputs and outputs
        self.X = data_train[:, 1:]
        self.y = data_train[:, 0]
        # self.X = df.values[:33,:-1]
        # self.y = df.values[:33,-1]
        # ensure input data is floats
        self.X = self.X.astype('float32')
        # label encode target and ensure the values are floats
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))
        


    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])

# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 50)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(50, 50)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # Third hidden layer
        # self.hidden3 = Linear(50, 100)
        # kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        # self.act3 = ReLU()
        # Fourth hidden layer and output
        self.hidden3 = Linear(50, 1)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
         # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        #third
        # X = self.hidden3(X)
        # X = self.act3(X)
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)
        return X

# prepare the dataset
def prepare_data(path):
    # load the dataset
    data = CSVDataset(path)
    # # calculate split
    # train, test = dataset.get_splits()
    # prepare data loaders
    data_dl = DataLoader(data, batch_size=100, shuffle=True)    
    # test_dl = DataLoader(test, batch_size=100, shuffle=False)
    return data_dl

# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=0.005, momentum=0.9)
    # enumerate epochs
    for epoch in range(100):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()

# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc

# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat

# prepare the data
path = 'train2D.txt'
path = 'train5D.txt'
train_dl = prepare_data(path)
# path = 'dev2D.txt'
# test_dl = prepare_data(path)
# print(len(train_dl.dataset), len(test_dl.dataset))
# define the network
model = MLP(5)
# train the model
print('Training Data...')
train_model(train_dl, model)
# evaluate the model
# print('Computing accuracy...')
# acc = evaluate_model(test_dl, model)
# print('Accuracy: %.3f' % acc)
# make a single prediction (expect class=1)
# yhat = []
# for i, (inputs, targets) in enumerate(test_dl):
#     prediction = model(inputs)
#     yhat = yhat.detach().numpy()
#     yhat = yhat.round()
#     predictions.append(yhat)
# # print('Predicted: %.3f (class=%d)' % (yhat, yhat.round()))
# print('Computing error rate')
# error = 0
# for input in range(test_dl.dataset.__len__()):
#     if (yhat[input] != test_dl.dataset.y[input]):
#         error += 1
# error_rate = float(error)/test_dl.dataset.__len__()
# print(error_rate)
###################################
#New way to predict
path = ['train2D.txt', 'dev2D.txt', 'eval2D.txt']
path = ['train5D.txt', 'dev5D.txt', 'eval5D.txt']
for data in path:
    f_test = open(data, 'r')
    data_test = np.genfromtxt(f_test)
    f_test.close()
    X = data_test[:, 1:]
    y = data_test[:, 0]
    predictions = []
    print('predicting data {}'.format(data))
    for i in range(len(X)):
        row = X[i]
        yhat = predict(row, model)
        predictions.append(yhat.round())
    if(data != 'eval5D.txt'):
        print('writing')
        writeResults(predictions, data)
    error = 0
    for input in range(len(y)):
        if (predictions[input] != y[input]):
            error += 1
    error_rate = float(error)/len(y)
    print(error_rate*100)

