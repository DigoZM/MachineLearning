# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:37:54 2020

@author: digo
"""
from pomegranate import *
#X = [['a','b',',a','c',',a','x','i'],['a','b','a','c','a','t','e']]
X = [[0, 1, 2, 1, 2, 1], [2, 1, 0, 1, 1, 2], [0, 1, 2, 2, 1, 0]]
model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=1, X=X)

Y = [2, 1, 0, 1, 1, 2]
result = model.log_probability(Y)
print(result)
Y = [2, 1, 0, 1, 1, 1]
result = model.log_probability(Y)
print(result)
Y = [2, 1, 0, 1, 1, 0]
result = model.log_probability(Y)
print(result)