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

    def clusters_distance():
        n = len(clusters)
        dc = np.zeros( (n,n), dtype = float )
        for c1 in range(n):
            for c2 in range(c1+1,n):
                dc1c2 = 0.
                for i in clusters[c1]:
                    for j in clusters[c2]:
                        dc1c2 += D[i,j]
                dc1c2 = dc1c2 / (1.*len(clusters[c1])*len(clusters[c2]))
                dc[c1,c2] = dc1c2
                dc[c2,c1] = dc1c2
        return dc    
    
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

    def merge_clusters_brute_force(c1,c2):
        c = clusters[c1]+clusters[c2]
        clusters.pop(c1)
        clusters.pop(c2-1)
        clusters.append(c)
        nid = max(clusterid)+1
        clusterid.pop(c1)        
        clusterid.pop(c2-1)
        clusterid.append(nid)
        return clusters_distance()
        
    def merge_clusters_to_be_fixed(c1,c2):
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
        for i in range(n):
            if i == c1 or i == c2:
                continue
            for j in range(n):
                if j == c1 or j == c2:
                    continue
                dmerged[ii,jj] = dclusters[i,j]
                jj +=1
            ii +=1
            jj = 0
        # update last column and row with merged cluster
        for i in range(n-2):
            dc = (dclusters[c1,i]*len(clusters[c1]) + dclusters[c2,i]*len(clusters[c2]))/float(len(clusters[c2])+len(clusters[c1]))
            dmerged[n-2,i] = dc
            dmerged[i,n-2] = dc
        c = clusters[c1]+clusters[c2]
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
        dclusters = merge_clusters_brute_force(c1,c2)
        cid = clusterid[-1]
        # add a new node C to T and connect it to nodes C1 and C2 by directed edges
        T[cid] = [cid1,cid2]
        T[cid1].append(cid)
        T[cid2].append(cid)        
        # Age(C) ← DC1,C2 / 2
        age[cid] = dc1c2/2.
#        print '--------------------------------'
#        print 'clusters',clusters
#        print 'clusterid',clusterid
#        print 'dclusters',dclusters
#        print 'clusters_distance',clusters_distance(clusters,D)
#        print 'age',age
#        print '--------------------------'        
    adj = {}
    for u,vs in T.iteritems():
        for v in vs:        
            w = abs(age[u] - age[v])
            adj.setdefault(u,[]).append((v,w))
    return adj
 
D = np.array([[0,20,17,11],[20,0,20,13],[17,20,0,10],[11,13,10,0]])
T = upgma(4,D)
res = {0: [(5, 7.0)], 1: [(6, 8.8333333333333339)], 2: [(4, 5.0)], 3: [(4, 5.0)], 4: [(2, 5.0), (3, 5.0), (5, 2.0)], 5: [(0, 7.0), (4, 2.0), (6, 1.8333333333333339)], 6: [(1, 8.8333333333333339), (5, 1.8333333333333339)]}
assert T == res


############################################################
#fpath = 'C:/Users/ngaude/Downloads/'
#fpath = '/home/ngaude/Downloads/'
fpath = 'C:/Users/Utilisateur/Downloads/'
############################################################

fname = fpath + 'dataset_10332_8.txt'
with open(fname, "r") as f:
    lines = f.read().strip().split('\n')
    n = int(lines[0])
    d = np.zeros(shape=(n,n),dtype = float)    
    for i,l in enumerate(lines[1:]):
        for j,v in enumerate(map(int,l.split('\t')[:-1])):
            d[i,j] = v
#    d = map(lambda r: map(int, r.split('\t')[:-1]), lines[1:])
#    d = np.array(d)
T = upgma(n,d)
with open(fname+'.out', "w") as f:
   for u,vw in T.iteritems():
       for (v,w) in vw:
           f.write(str(u)+'->'+str(v)+':'+"{0:.3f}".format(w)+'\n')