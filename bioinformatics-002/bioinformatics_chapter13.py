# -*- coding: utf-8 -*-
"""
Created on Sat May 09 23:12:54 2015

@author: ngaude
"""

import math

def distance(a,b):
    # given a,b points as a two tuples, returns distance a,b
    return math.sqrt(sum([ (ai-bi)*(ai-bi) for ai,bi in zip(a,b)]))
    

def max_distant(c,dp):
    # given data points dp and centers c
    # returns the point from data points that maximise the distance to centers
    dpd = [(min([distance(p,ci) for ci in c]),p) for p in dp]
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
        c.append(max_distant(c,dp))
    return c

def closest_center(c,p):
    # given centers c and point p
    # return the closest center from p
    dp = [(distance(ci,p),ci) for ci in c]
    return min(dp)[1]
    

k = 3
dp = [(0.0, 0.0),(5.0, 5.0),(0.0, 5.0),(1.0, 1.0),(2.0, 2.0),(3.0, 3.0),(1.0, 2.0)]   
assert set(farthest_first_traversal(k,dp)) == set([(0.0, 0.0), (5.0, 5.0), (0.0, 5.0)])
    
def squared_error_distortion(c,dp):
    """
    CODE CHALLENGE: Solve the Squared Error Distortion Problem.
    Input: Integers k and m, followed by a set of centers Centers and a set of points Data.
    Output: The squared error distortion Distortion(Data, Centers).
    """
    return 1./len(dp)*sum([math.pow(distance(p,closest_center(c,p)),2) for p in dp])
    
c = [(2.31, 4.55),(5.96, 9.08)]
dp = [(3.42, 6.03),(6.23, 8.25),(4.76, 1.64),(4.47, 4.33),(3.95, 7.61),(8.93, 2.97),(9.74, 4.03),(1.73, 1.28),(9.72, 5.01),(7.27, 3.77)]
assert abs(squared_error_distortion(c,dp) - 18.246) < 0.001

def lloyd_kmeans(k,dp,eps = 0.001):
    """
    CODE CHALLENGE: Implement the Lloyd algorithm for k-means clustering.
    Input: Integers k and m followed by a set of points Data in m-dimensional space.
    Output: A set Centers consisting of k points (centers) resulting from applying the
    Lloyd algorithm to Data and Centers, where the first k points from Data are selected
    as the first k centers.
    """
    c = dp[:k]
    err = squared_error_distortion(c,dp)
    derr = 1
    def centroid(pp):
        return tuple([1.*sum(ai)/len(ai) for ai in zip(*pp)])
    while derr > eps:
        cluster = {}
        for p in dp:
            ci = closest_center(c,p)
            cluster.setdefault(ci,[]).append(p)
        for i,ci in enumerate(c):
            c[i] = centroid(cluster[ci])
        perr = err
        err = squared_error_distortion(c,dp)
        derr = abs(err-perr)
    return c

k = 2
dp = [(1.3, 1.1),(1.3, 0.2),(0.6, 2.8),(3.0, 3.2),(1.2, 0.7),(1.4, 1.6),(1.2, 1.0),(1.2, 1.1),(0.6, 1.5),(1.8, 2.6),(1.2, 1.3),(1.2, 1.0),(0.0, 1.9)]
c = lloyd_kmeans(k,dp,eps = 0.001)
assert abs(c[0][0] - 1.800) < 0.001
assert abs(c[0][1] - 2.866) < 0.001
assert abs(c[1][0] - 1.059) < 0.001
assert abs(c[1][1] - 1.140) < 0.001



        
    

############################################################
fpath = 'C:/Users/ngaude/Downloads/'
#fpath = '/home/ngaude/Downloads/'
#fpath = 'C:/Users/Utilisateur/Downloads/'
############################################################

#fname = fpath + 'FarthestFirstTraversal.txt'
#fname = fpath + 'dataset_10926_14.txt'
#with open(fname, "r") as f:
#    lines = f.read().strip().split('\n')
#    k = int(lines[0].split(' ')[0])
#    dp = map(lambda l:tuple(map(float,l.split(' '))),lines[1:])
#    c = farthest_first_traversal(k,dp)
#with open(fname+'.out', "w") as f:
#    for ci in c:
#        f.write(' '.join(map(str,ci))+'\n')

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