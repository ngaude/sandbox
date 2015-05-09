# -*- coding: utf-8 -*-
"""
Created on Sat May 09 23:12:54 2015

@author: ngaude
"""

import math

def distance(a,b):
    """
    given a,b points as a two tuples, returns distance a,b
    """
    return math.sqrt(sum([ (ai-bi)*(ai-bi) for ai,bi in zip(a,b)]))
    

def max_distant(dp,c):
    """
    given data points dp and centers c
    returns the point from data points that maximise the distance to centers
    """
    dpd = [(sum([distance(p,ci) for ci in c]),p) for p in dp]
    return max(dpd)[1]

def farthest_first_traversal(k,dp):
    """
    CODE CHALLENGE: Implement the FarthestFirstTraversal clustering heuristic.
    Input: Integers k and m followed by a set of points Data in m-dimensional space.
    Output: A set Centers consisting of k points (centers) resulting from applying
    FarthestFirstTraversal(Data, k), where the first point from Data is chosen as the
    first center to initialize the algorithm.
    """
    c = [dp[0],]
    while len(c)<k:
        c.append(max_distant(dp,c))
    return c
    
k = 3
dp = [(0.0, 0.0),(5.0, 5.0),(0.0, 5.0),(1.0, 1.0),(2.0, 2.0),(3.0, 3.0),(1.0, 2.0)]        
assert set(farthest_first_traversal(k,dp)) == set([(0.0, 0.0), (5.0, 5.0), (0.0, 5.0)])

############################################################
fpath = 'C:/Users/ngaude/Downloads/'
#fpath = '/home/ngaude/Downloads/'
#fpath = 'C:/Users/Utilisateur/Downloads/'
############################################################


    