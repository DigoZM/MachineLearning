# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:37:54 2020

@author: digo
"""
from pomegranate import *
#X = [['a','b',',a','c',',a','x','i'],['a','b','a','c','a','t','e']]
NM = ord('0')
PC = ord('.')
alphabet = 'abcdefghijklmnopqrstuvwxyz'

def WordIntoList(word):
    res = []
    
    for i in range (len(word)):
        res.append(ord(word[i]))
        
    return res

def PredictNext (word):
    Y = WordIntoList(word)
    Y.append('a')
    next_letter = len(Y) - 1
    maximum_prob = -10000000000
    result = 'nothing'
    order = []
    for letter in range (len(alphabet)):
        Y[next_letter] = ord(alphabet[letter])
        print(Y)
        temp_prob = model.summarize([Y,Y])
#        print(temp_prob)
        order.append([( temp_prob, alphabet[letter])])
        if(temp_prob > maximum_prob):
            maximum_prob = temp_prob
            result = alphabet[letter]
    return order
        
    
    
    
    
    
#Reading File
##Train data
f_train = open('data.txt', 'r')
data = f_train.read()
f_train.close()
data_train = data.split()
data_train = map(str, data_train)
data_train_numeral = []
for i in data_train:
    aux = []
#    print(i)
    for j in range(len(i)):
        character = i[j].lower()
        if (character >= 'a' and character <= 'z'):
            aux.append(ord(character))
        elif (character >= '0' and character <= '9'):
            aux.append(NM)
        else:
            aux.append(PC)
    data_train_numeral.append(aux)
    

train_data = []
X = [[0, 1, 2, 1, 2, 1], [2, 1, 0, 1, 1, 2], [0, 1, 2, 2, 1, 0]]
model = HiddenMarkovModel()
#aaa =NormalDistribution(1, 7)
#s1 = State(NormalDistribution(1, 7))
#s2 = State(NormalDistribution(1, 7))
#s3 = State(NormalDistribution(8, 2))
#model.add_transition(model.start, s1, 1.0)
#model.add_transition(s1, s1, 0.7)
#model.add_transition(s1, s2, 0.3)
#model.add_transition(s2, s2, 0.8)
#model.add_transition(s2, s3, 0.2)
#model.add_transition(s3, s3, 0.9)
#model.add_transition(s3, model.end, 0.1)
#model.add_states(s1, s2, s3)
#model.bake()
#model.fit(data_train_numeral, algorithm='viterbi')
model = HiddenMarkovModel.from_samples( n_components=5, X=data_train_numeral)
X.append([0, 1, 2, 1, 2, 1])
Y = 'though'
result0 = PredictNext(Y)
result0.sort(key = lambda x:x[0], reverse = True)
Y = 'comin'
result1 = PredictNext(Y)
result1.sort(key = lambda x:x[0], reverse = True)
Y = 'whatever'
result2 = PredictNext(Y)
result2.sort(key = lambda x:x[0], reverse = True)
