# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:27:45 2015

@author: ngaude
"""

import numpy as np
import math

def distance(a,b):
    # given a,b points as a two tuples, returns distance a,b
    return math.sqrt(sum([ (ai-bi)*(ai-bi) for ai,bi in zip(a,b)]))

def centroid(w,dp):
        ww = sum(w)
        pp = [ [1.*x*wi/ww for x in xs] for xs,wi in zip (dp,w)]
        return tuple([sum(ai) for ai in zip(*pp)])
        
def hidden_matrix(c,b,dp):
    # given data points dp and centers c
    # returns the hidden matrix of 
    hm = np.array([[math.exp(-b*distance(xi,ci)) for xi in dp] for ci in c])
    return hm/hm.sum(axis=0)
    
dp = ((0,1),(5,6),(4,5))
b = 1
c = ((0,0),(3,3))
hm = hidden_matrix(c,b,dp)

def soft_kmeans(k,b,dp):
    """
    CODE CHALLENGE: Implement the expectation maximization algorithm for soft k-means clustering.
    Input: Integers k and m, followed by a stiffness parameter β, followed by a set of points
    Data in m-dimensional space.
    Output: A set Centers consisting of k points (centers) resulting from applying the
    expectation maximization algorithm for soft k-means clustering. Select the first k points
    from Data as the first centers for the algorithm and run the algorithm for 100 E-steps
    and 100 M-steps. Results should be accurate up to three decimal places.
    """
    c = dp[:k]
    for i in range(100):
        hm = hidden_matrix(c,b,dp)
        c = [centroid(hmi,dp) for hmi in hm]
    return c
    
k = 2
b = 2.7
dp = [(1.3, 1.1),(1.3, 0.2),(0.6, 2.8),(3.0, 3.2),(1.2, 0.7),(1.4, 1.6),(1.2, 1.0),(1.2, 1.1),(0.6, 1.5),(1.8, 2.6),(1.2, 1.3),(1.2, 1.0),(0.0, 1.9)]
c = soft_kmeans(k,b,dp)

assert map(lambda t:map(lambda f:round(f,3),t),c) == [[1.662, 2.623], [1.075, 1.148]]


############################################################
fpath = 'C:/Users/ngaude/Downloads/'
#fpath = '/home/ngaude/Downloads/'
#fpath = 'C:/Users/Utilisateur/Downloads/'
############################################################

fname = fpath + 'dataset_10933_7.txt'
with open(fname, "r") as f:
    lines = f.read().strip().split('\n')
    k = int(lines[0].split(' ')[0])
    b = float(lines[1])
    dp = map(lambda l:tuple(map(float,l.split(' '))),lines[2:])
    c = soft_kmeans(k,b,dp)
with open(fname+'.out', "w") as f:
    for ci in c:
        f.write(' '.join(map(str,ci))+'\n')

#fname = fpath + 'SquaredErrorDistortion.txt'
#fname = fpath + 'dataset_10927_3.txt'
#with open(fname, "r") as f:
#    lines = f.read().strip().split('\n')
#    k = int(lines[0].split(' ')[0])
#    c = map(lambda l:tuple(map(float,l.split(' '))),lines[1:k+1])
#    dp = map(lambda l:tuple(map(float,l.split(' '))),lines[k+2:])
#print squared_error_distortion(c,dp)

#fname = fpath + 'Lloyd.txt'
#fname = fpath + 'dataset_10928_3.txt'
#with open(fname, "r") as f:
#    lines = f.read().strip().split('\n')
#    k = int(lines[0].split(' ')[0])
#    dp = map(lambda l:tuple(map(float,l.split(' '))),lines[1:])
#    c = lloyd_kmeans(k,dp)
#with open(fname+'.out', "w") as f:
#    for ci in c:
#        f.write(' '.join(map(str,ci))+'\n')    