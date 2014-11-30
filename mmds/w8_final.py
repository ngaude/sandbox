# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 23:33:40 2014

@author: Utilisateur
"""


import numpy as np
import math

B = np.array([0,3,1,2,0])
D = np.array([0,4,3,0,2])

eB = B-np.mean(B)
eD = D-np.mean(D)

eB = np.array([0,1,-1,0,0])
eD = np.array([0,1,0,0,-1]) 

dD = math.sqrt(np.dot(eD,eD))
dB = math.sqrt(np.dot(eB,eB)) 
p = np.dot(eB,eD)/(dB*dD)


print p