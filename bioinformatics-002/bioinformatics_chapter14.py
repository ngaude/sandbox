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
    if (b == None):
        hm = np.array([[1/math.pow(distance(xi,ci),2) for xi in dp] for ci in c])
    else:
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


def cluster_distance(ca,cb,dm):
    dab = 0.
    for i in ca:
        for j in cb:
            dab += dm[i-1,j-1]
    dab = dab / (1.*len(ca)*len(cb))
    return dab

def hierarchical_clustering(n,dm):
    """
    CODE CHALLENGE: Implement HierarchicalClustering.
    Input: An integer n, followed by an n x n distance matrix.
    Output: The result of applying HierarchicalClustering to this distance matrix (using Davg),
    with each newly created cluster listed on each line.
    """
    c = [ [1+i,] for i in range(n)]
    r = []
    odm = dm
    while len(dm)>2:
        n = len(dm)
        a,b = np.unravel_index(np.argmin(dm + np.eye(n)*dm.max()),(n,n))
        ca = c[a][:]
        cb = c[b][:]
        cc = ca+cb      
        c.remove(ca)
        c.remove(cb)
        c.append(cc)
        dm = np.delete(dm,(a,b),0)
        dm = np.delete(dm,(a,b),1)
        ndm = np.zeros((n-1,n-1))
        ndm[:-1,:-1] = dm
        for i in range(n-2):
            dicc = cluster_distance(c[i],cc,odm)
            ndm[-1,i] = dicc
            ndm[i,-1] = dicc        
        r.append(cc)
        dm = ndm
    r.append(c[0]+c[1])
    return r
        
        

dm = ((0.00, 0.74, 0.85, 0.54, 0.83, 0.92, 0.89),
      (0.74, 0.00, 1.59, 1.35, 1.20, 1.48, 1.55),
      (0.85, 1.59, 0.00, 0.63, 1.13, 0.69, 0.73),
      (0.54, 1.35, 0.63, 0.00, 0.66, 0.43, 0.88),
      (0.83, 1.20, 1.13, 0.66, 0.00, 0.72, 0.55),
      (0.92, 1.48, 0.69, 0.43, 0.72, 0.00, 0.80),
      (0.89, 1.55, 0.73, 0.88, 0.55, 0.80, 0.00))
dm = np.array(dm)
r = hierarchical_clustering(len(dm),dm)
assert r == [[4, 6], [5, 7], [3, 4, 6], [1, 2], [5, 7, 3, 4, 6], [1, 2, 5, 7, 3, 4, 6]]


############################################################
fpath = 'C:/Users/ngaude/Downloads/'
#fpath = '/home/ngaude/Downloads/'
#fpath = 'C:/Users/Utilisateur/Downloads/'
############################################################

#fname = fpath + 'dataset_10933_7.txt'
#with open(fname, "r") as f:
#    lines = f.read().strip().split('\n')
#    k = int(lines[0].split(' ')[0])
#    b = float(lines[1])
#    dp = map(lambda l:tuple(map(float,l.split(' '))),lines[2:])
#    c = soft_kmeans(k,b,dp)
#with open(fname+'.out', "w") as f:
#    for ci in c:
#        f.write(' '.join(map(str,ci))+'\n')

#fname = fpath + 'dataset_10934_7.txt'
##fname = fpath + 'HierarchicalClustering.txt'
#with open(fname, "r") as f:
#    lines = f.read().strip().split('\n')
#    n = int(lines[0].split(' ')[0])
#    dm = map(lambda l:map(float,l.split(' ')),lines[1:])
#    dm = np.array(dm)
#    r = hierarchical_clustering(n,dm)
#with open(fname+'.out', "w") as f:
#    for ri in r:
#        f.write(' '.join(map(str,ri))+'\n')

############################################################
# QUIZZ
############################################################

# θ = 0.6. Compute Pr(HHTHHH|θ)
p = math.pow(0.6,5)*math.pow(0.4,1)
print 'p=',round(p,3)

dp = ((2,8), (2,5), (6,9), (7,5), (5,2))
c = ((3,5), (5,4))
b = 1
hm = hidden_matrix(c,b,dp)
print 'hm[1,2]=',round(hm[0,1],3)

dp = ((2,8), (2,5), (6,9), (7,5), (5,2))
w = (0.5, 0.3, 0.8, 0.4, 0.9)
c = centroid(w,dp)
print 'c=',' '.join(map(lambda f:str(round(f,3)),c))

#   i  j  k  l
#i  0 20  9 11
#j 20  0 17 11
#k  9 17  0  8
#l 11 11  8  0 
dm = ((0, 20, 9, 11),(20, 0, 17, 11),(9, 17,  0,  8),(11, 11,  8,  0)) 
dm = np.array(dm)
print 'cluster_distance=',cluster_distance((1,3),(2,4),dm)
