# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 17:13:27 2020

@author: digo
"""
def CompareString(first, second):
    size = min(len(first), len(second))
    error = abs(len(first), len(second))
    for i in range (size):
        if(first[i] != second[i]):
            error += 1
    return error
        

    

reference_txt = "Machine Learning gives computers the ability to learn without being explicitly programmed."
hypotesis_txt = "Deep Learning allows computers to learn without they are explicitly codded."
#reference_txt = 'a b b c d'
#hypotesis_txt = 'a c c d'
reference_txt.lower()
hypotesis_txt.lower()

#Arrangin Data
aux = []
reference = []
hypotesis = []
for i in reference_txt:
    if(i != ' '):
        aux.append(i)
    else:
        reference.append(aux)
        aux = []
reference.append(aux)
aux = []
for i in hypotesis_txt:
    if(i != ' '):
        aux.append(i)
    else:
        hypotesis.append(aux)
        aux = []
hypotesis.append(aux)

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
#            print('iguais')
#            print(reference[j-1])
        else:
#            subs = CompareString(reference[j-1], hypotesis[i-1])
            subs = 1
        substitution = table[i-1][j-1] + subs
        insertion = table[i][j-1] + subs
        deletion = table[i-1][j]  + subs
        table[i][j] = min(substitution, insertion, deletion)
     
#Result     
#print(table[i][j])
result = table[i][j]
#Recovering
SUBS = '*'
ADD = '+'
DEL = '-'
subs = 0
add = 0
deletion = 0
recoverd = []
i = len(hypotesis)
j = len(reference)
while((i > 0) or (j > 0)):
    if(i > 0 and ((table[i][j] > table[i-1][j]) or (table[i][j] > table[i-1][j-1]))):
        if(table[i][j] > table[i-1][j]):
            for k in range (len(hypotesis[i-1])):
                recoverd.insert(0, DEL)
            i -= 1
            deletion += 1
        elif(table[i][j] > table[i-1][j-1]):
            for k in range (len(reference[j-1])):
                recoverd.insert(0, SUBS)
            j -= 1
            i -= 1
            subs += 1
    elif(j > 0 and table[i][j] > table[i][j-1]):
        for k in range(len(reference[j-1])):        
            recoverd.insert(0, ADD)
        j -= 1
        add += 1
    else:
        for k in range(len(reference[j-1])):
            recoverd.insert(k, reference[j-1][k])
        j -= 1
        i -= 1
    recoverd.insert(0, ' ')
recoverd.pop(0)
print("\n")
print("R: {}".format(reference_txt))        
recoverd = ''.join(map(str, recoverd))
print("H: {}".format(hypotesis_txt))  
print("Answer:")
print(reference_txt)
print(recoverd)
print("Subs: {} Ins: {} Del: {} Total Errors: {}".format(subs, add, deletion, result))     
print('\n')
#recoverd = []
#i = len(hypotesis)
#j = len(reference)
#while((i > 0) and (j > 0)):
#    if(table[i][j] > table[i-1][j-1]):
#        recoverd.insert(0, SUBS)
#        j -= 1
#        i -= 1
#    elif(table[i][j] > table[i][j-1]):
#        recoverd.insert(0, ADD)
#        j -= 1
#    elif(table[i][j] > table[i-1][j]):
#        recoverd.insert(0, DEL)
#        i -= 1 
#    else:
#        recoverd.insert(0, reference[j-1])
#        j -= 1
#        i -= 1 
#
#print(reference)        
#recoverd = ''.join(map(str, recoverd))
#print(recoverd)
#print(hypotesis)











