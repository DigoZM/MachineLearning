# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 12:22:34 2020

@author: digo
"""
import numpy as np

#Generate Points
mu = 1
sigma = 1
s = np.random.normal(mu, sigma, 1000000)

#Write data in file

file_object  = open('set10.txt', 'w')
for i in range (len(s)):
    file_object.write('%.17f\n' % s[i])

