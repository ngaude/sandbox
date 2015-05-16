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


def neighbor_joining_matrix(D):
    n = len(D)
    tD = D.sum(axis=1)      
    tD.shape = (n,1)
    o = np.ones(n)
    o.shape = (1,n)
    tD = np.dot(tD,o)
    njD = (n-2.)*D - tD - tD.transpose()
    for i in range(n):
        njD[i,i] = 0
    return njD

def neighbor_joining(n,D):
    """
    CODE CHALLENGE: Implement NeighborJoining.
    Input: An integer n, followed by an n x n distance matrix.
    Output: An adjacency list for the tree resulting from applying 
    the neighbor-joining algorithm.
    """
    def total_distance():
        return D.sum(axis=1)
    
    def neighbor_joining_distance():
        n = len(D)
        tD = total_distance()        
        tD.shape = (n,1)
        o = np.ones(n)
        o.shape = (1,n)
        tD = np.dot(tD,o)
        njD = (n-2.)*D - tD - tD.transpose()
        for i in range(n):
            njD[i,i] = 0
        return njD

    def find_pair():
        n = len(D)
        assert n>2
        njD = neighbor_joining_distance()
        dmin = njD[0,1] 
        pmin = (0,1)                
        for i in range(n):            
            for j in range(i+1,n):
                if njD[i,j] < dmin:
                    dmin = njD[i,j]
                    pmin = (i,j)
        return pmin

    def delta_distance():
        n = len(D)
        tD = total_distance()        
        tD.shape = (n,1)
        o = np.ones(n)
        o.shape = (1,n)
        tD = np.dot(tD,o)
        return (tD-tD.transpose())/float(n-2)
    
    def limb_length(i,j):
        dD = delta_distance()
        lli = 0.5*(D[i,j]+dD[i,j])
        llj = 0.5*(D[i,j]-dD[i,j])
        return (lli,llj)

    def reduce_distance(c1,c2):
        n = len(D)
        r = range(n)
        # add a new row/column m to D 
        # so that Dk,m = Dm,k = (1/2)(Dk,i + Dk,j - Di,j) for any k
        aD = np.zeros( (n+1,n+1), dtype = float )
        for ii,i in enumerate(r):
            for jj,j in enumerate(r):
                aD[ii,jj] = D[i,j]
        # do not compute m row/colmun for c1 row/column as well as c2 
        r.remove(c1)
        r.remove(c2)
        for k in r:
            aDk = 0.5*(D[k,c1]+D[k,c2]-D[c1,c2])
            aD[k,n] = aDk
            aD[n,k] = aDk
        # remove c1 row/column as well as c2 row/column from D/aD
        rD = np.zeros( (n-1,n-1), dtype = float )
        # copy also the m row/column from aD
        r.append(n)
        for ii,i in enumerate(r):
            for jj,j in enumerate(r):
                rD[ii,jj] = aD[i,j]
        return rD
    T = {}
    nodeid = range(n) 
    while len(D)>2:
        (i,j) = find_pair()
        (ii,jj) = (nodeid[i],nodeid[j])
        (wi,wj) = limb_length(i,j)
        # insert a k node linked to i and j
        kk =  max(nodeid)+1
        T.setdefault(kk,[]).append((ii,wi))
        T.setdefault(ii,[]).append((kk,wi))
        T.setdefault(kk,[]).append((jj,wj))
        T.setdefault(jj,[]).append((kk,wj))

        # update the node id reference
        nodeid.append(kk)
        nodeid.remove(ii)
        nodeid.remove(jj)
#        print '------'
#        print 'D',D
#        print 'njD',neighbor_joining_distance()
#        print 'tD',total_distance()
#        print '(i,j)',(ii,jj),'(w,w)',wi,wj
#        print 'nodeid',nodeid
#        print tree_tostring(T)
#        print '------'
    
        # reduce the distance matrix
        D = reduce_distance(i,j)

        
    # insert the last edge from the D 2x2 matrix
    T.setdefault(nodeid[0],[]).append((nodeid[1],D[0,1]))
    T.setdefault(nodeid[1],[]).append((nodeid[0],D[1,0]))
    
    return T

def tree_tostring(T):
    s = ''
    for u,vw in T.iteritems():
        for (v,w) in vw:
            s += str(u)+'->'+str(v)+':'+"{0:.3f}".format(w)+'\n'
    return s

T = neighbor_joining(4,D = np.array([[0,13,21,22],[13,0,12,13],[21,12,0,13],[22,13,13,0]]))
res = {0: [(4, 11.0)], 1: [(4, 2.0)], 2: [(5, 6.0)], 3: [(5, 7.0)], 4: [(0, 11.0), (1, 2.0), (5, 4.0)], 5: [(2, 6.0), (3, 7.0), (4, 4.0)]}
assert res == T

T = neighbor_joining(4,D = np.array([[0,3,4,3],[3,0,4,5],[4,4,0,2],[3,5,2,0]]))
res = {0: [(4, 1.0)], 1: [(4, 2.0)], 2: [(5, 1.0)], 3: [(5, 1.0)], 4: [(0, 1.0), (1, 2.0), (5, 1.5)], 5: [(2, 1.0), (3, 1.0), (4, 1.5)]}
assert res == T

T = neighbor_joining(4,D = np.array([[0,23,27,20],[23,0,30,28],[27,30,0,30],[20,28,30,0]],dtype = float))
res = {0: [(4, 8.0)], 1: [(5, 13.5)], 2: [(5, 16.5)], 3: [(4, 12.0)], 4: [(0, 8.0), (3, 12.0), (5, 2.0)], 5: [(1, 13.5), (2, 16.5), (4, 2.0)]}
assert res == T

############################################################
#fpath = 'C:/Users/ngaude/Downloads/'
#fpath = '/home/ngaude/Downloads/'
fpath = 'C:/Users/Utilisateur/Downloads/'
############################################################

#fname = fpath + 'dataset_10332_8.txt'
#with open(fname, "r") as f:
#    lines = f.read().strip().split('\n')
#    n = int(lines[0])
#    d = np.zeros(shape=(n,n),dtype = float)    
#    for i,l in enumerate(lines[1:]):
#        for j,v in enumerate(map(int,l.split('\t')[:-1])):
#            d[i,j] = v
#T = upgma(n,d)
#with open(fname+'.out', "w") as f:
#   for u,vw in T.iteritems():
#       for (v,w) in vw:
#           f.write(str(u)+'->'+str(v)+':'+"{0:.3f}".format(w)+'\n')

#fname = fpath + 'dataset_10333_6.txt'
#with open(fname, "r") as f:
#    lines = f.read().strip().split('\n')
#    n = int(lines[0])
#    d = np.zeros(shape=(n,n),dtype = float)    
#    for i,l in enumerate(lines[1:]):
#        l = l.replace('\t',' ')
#        valstr = l.split(' ')
#        if valstr[-1]=='':
#            valstr = valstr[:-1]
#        for j,v in enumerate(map(int,valstr)):
#            d[i,j] = v
#T = neighbor_joining(n,d)
#with open(fname+'.out', "w") as f:
#    f.write(tree_tostring(T))

############################################################
# QUIZZ
############################################################

D = np.array([[0,13,16,10],[13,0,21,15],[16,21,0,18],[10,15,18,0]])
print neighbor_joining_matrix(D)

