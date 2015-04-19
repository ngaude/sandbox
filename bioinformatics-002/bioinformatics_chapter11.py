# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 23:23:18 2015

@author: Utilisateur
"""

import numpy as np

def upgma(n,D):
    """
    CODE CHALLENGE: Implement UPGMA.
    Input: An integer n followed by a space separated n x n distance matrix.
    Output: An adjacency list for the ultrametric tree returned by UPGMA. Edge weights
    should be accurate to three decimal places.
    """
    
    def find_closest_clusters():
        (n,m) = dclusters.shape
        assert n == m
        assert n > 1
        dmin = dclusters[0,1] 
        pmin = (0,1)                
        for i in range(n):            
            for j in range(i+1,n):
                if dclusters[i,j] < dmin:
                    dmin = dclusters[i,j]
                    pmin = (i,j)
        return pmin
        
    def merge_clusters(c1,c2):
        # return a new distance cluster matrix and update the clusters list...
        (n,m) = dclusters.shape
        assert n == m
        assert c1 != c2
        assert c1 < n
        assert c2 < m
        # copy all element but c1,c2 columns,rows
        ii = 0
        jj = 0
        dmerged = np.zeros( (n-1,n-1), dtype = float )
        print 'n-1',n-1
        for i in range(n):
            if i == c1 or i == c2:
                continue
            for j in range(n):
                if j == c1 or j == c2:
                    continue
                print ii,jj
                dmerged[ii,jj] = dclusters[i,j]
                jj +=1
            ii +=1
            jj = 0
        # update last column and row with merged cluster
        for i in range(n-1):
            dc = (dclusters[c1,i]*len(clusters[c1]) + dclusters[c2,i]*len(clusters[c2]))/(len(clusters[c2])+len(clusters[c1]))
            dmerged[n-2,i] = dc
            dmerged[i,n-2] = dc
        c = clusters[c1]+clusters[c2]
        print 'clusters',clusters
        print 'c1',c1
        print 'c2',c2
        clusters.pop(c1)
        clusters.pop(c2-1)
        clusters.append(c)
        nid = max(clusterid)+1
        clusterid.pop(c1)        
        clusterid.pop(c2-1)
        clusterid.append(nid)
        return dmerged        

    # init vars    
    dclusters = D
    T = {}
    clusters = []
    clusterid = []
    age = {}    
    for i in range(n):
        clusters.append([i])
        clusterid.append(i) 
        T[i] = []
        age[i] = 0
    
    # iterate
    while len(clusters)>1:
        # find two closest clusters C1 and C2 (break ties arbitrarily)
        (c1,c2) = find_closest_clusters()
        cid1 = clusterid[c1]
        cid2 = clusterid[c2]        
        dc1c2 = dclusters[c1,c2]
        # merge C1 and C2 into a new cluster C
        dclusters = merge_clusters(c1,c2)
        cid = clusterid[-1]
        # add a new node C to T and connect it to nodes C1 and C2 by directed edges
        T[cid] = [cid1,cid2]
        T[cid1].append(cid)
        T[cid2].append(cid)        
        # Age(C) ← DC1,C2 / 2
        age[cid] = dc1c2/2
    adj = {}
    print 'T',T
    for u,vs in T.iteritems():
        for v in vs:        
            w = age[u] - age[v]
            adj.setdefault(u,[]).append((v,w))
    return adj
"""    
    UPGMA(D, n)
 form n clusters, each containing a single element i (for i from 1 to n)
 construct a graph T by assigning a node to each cluster (without adding any edges) 
 for every node v in T 
  Age(v) ← 0
 while there is more than one cluster
  find two closest clusters C1 and C2 (break ties arbitrarily)
  merge C1 and C2 into a new cluster C
  add a new node C to T and connect it to nodes C1 and C2 by directed edges
  Age(C) ← DC1,C2 / 2
  remove rows and columns of D corresponding to C1 and C2
  add a row and column to D for C by recomputing DC,C* for each C* ≠ C
 root ← the node in T corresponding to the cluster C
 for each edge (v,w) in T
  Length(v,w) ← Age(v) - Age(w)
 return T
 """
 
D = np.array([[0,20,17,11],[20,0,20,13],[17,20,0,10],[11,13,10,0]])
adj = upgma(4,D)
print adj