# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 17:13:27 2020

@author: digo
"""


    

reference = "See Jane and Mary run up the hill"
hypotesis = "Jerry and Mike run quickly to the mill"
#reference = 'abbcd'
#hypotesis = 'accd'
reference.lower()
hypotesis.lower()


#######Dynamic Programming Algorithm######
#setting table 
table = []
aux = []
for i in range(len(hypotesis) + 1):
    for j in range(len(reference) + 1):
        aux.append(0)
    table.append(aux)
    aux = []
#Boundary Conditions
table[0][0] = 0
for i in range (len(hypotesis) + 1):
    table[i][0] = i
for j in range (len(reference) + 1):
    table[0][j] = j
#Computing the cost
for i in range(1, len(hypotesis) + 1):
    for j in range(1, len(reference) + 1):
        if(reference[j-1] == hypotesis[i-1]):
            subs = 0
        else:
            subs = 1
        substitution = table[i-1][j-1] + subs
        insertion = table[i][j-1] + 1
        deletion = table[i-1][j]  + 1
        table[i][j] = min(substitution, insertion, deletion)
     
#Result     
print(table[i][j])

#Recovering
SUBS = '*'
ADD = '+'
DEL = '-'
recoverd = []
i = len(hypotesis)
j = len(reference)
while((i > 0) and (j > 0)):
    if(table[i][j] > table[i][j-1]):
        recoverd.insert(0, ADD)
        j -= 1
    elif(table[i][j] > table[i-1][j]):
        recoverd.insert(0, DEL)
        i -= 1 
    elif(table[i][j] > table[i-1][j-1]):
        recoverd.insert(0, SUBS)
        j -= 1
        i -= 1
    else:
        recoverd.insert(0, reference[j-1])
        j -= 1
        i -= 1 
print(reference)        
recoverd = ''.join(map(str, recoverd))
print(recoverd)
print(hypotesis)       
print('\n')
recoverd = []
i = len(hypotesis)
j = len(reference)
while((i > 0) and (j > 0)):
    if(table[i][j] > table[i-1][j-1]):
        recoverd.insert(0, SUBS)
        j -= 1
        i -= 1
    elif(table[i][j] > table[i][j-1]):
        recoverd.insert(0, ADD)
        j -= 1
    elif(table[i][j] > table[i-1][j]):
        recoverd.insert(0, DEL)
        i -= 1 
    else:
        recoverd.insert(0, reference[j-1])
        j -= 1
        i -= 1 

print(reference)        
recoverd = ''.join(map(str, recoverd))
print(recoverd)
print(hypotesis)











