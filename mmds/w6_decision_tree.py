# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 17:50:41 2014

@author: ngaude
"""


 

def dtree(x):
    a = x[0]
    s = x[1]
    if (a <45):
        if (s < 110): return False
        else: return True
    else:
        if (s < 75): return False
        else: return True

gold = ((28,145), (38,115), (43,83), (50,130), (50,90), (50,60), (50,30), (55,118), (63,88), (65,140))
nogold = ((23,40), (25,125), (29,97), (33,22), (35,63), (42,57), (44, 105), (55,63), (55,20), (64,37))

for t in nogold:
    if dtree(t) == True:
        print t,'misclassified'

for t in gold:
    if dtree(t) == False:
        print t,'misclassified'
