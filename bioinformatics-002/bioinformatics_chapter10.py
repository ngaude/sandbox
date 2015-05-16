# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:14:04 2015

@author: ngaude
"""

import numpy as np

def knight_detour():
    def adjacency_list(n,e):
        adj = {}
        for edge in e:
            adj.setdefault(edge[1],[]).append(edge[0])
            adj.setdefault(edge[0],[]).append(edge[1])
        return adj
    def move(path):
        if (len(path) == len(nodes)):
            #eulerian path found
            if path[0] in adj[path[-1]]:
                #eulerian cycle found
                tour.append(path)
            return
        for n in adj[path[-1]]:
            if n in path:
                continue
            npath = path[:]
            npath.append(n)
            move(npath)        
    nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    edges = [(1,6), (1,7), (1,9), (2,3), (2,8), (2,10), (3,9), (3,11), (4,10), (4,12), (5,7), (5,11), (6,8), (6,12), (7,12), (10,11)]
    adj = adjacency_list(nodes,edges)
    tour = []
    move([1,])
    print '\n'.join(map(str,tour))
#    return

def distances_between_leaves(n,adjacency_list):
    """    
    Distances Between Leaves Problem: Compute the distances between leaves in a weighted tree.
    Input:  An integer n followed by the adjacency list of a weighted tree with n leaves.
    Output: An n x n matrix (di,j), where di,j is the length of the path between leaves i and j.
    """
    adj = {}
    for (k, v, w) in adjacency_list:
        adj.setdefault(k,[]).append((v,w))
        
    # assume the keys are given from [0,....,p]
    assert min(adj.keys()) == 0
    assert max(adj.keys()) == len(adj.keys()) - 1
    leaves = [k for k,v in adj.iteritems() if len(v) == 1]
    assert len(leaves) == n
    node_to_node_distance = np.zeros( (len(adj),len(adj)), dtype = int )
    leaf_to_leaf_distance = np.zeros( (n,n), dtype = int )   
    
    def dfs(path):
        for v,w in adj[path[-1][0]]:
            if visited[v] == True:
                continue
            visited[v] = True
            pathlen = path[-1][1] + w
            for (iv,ilen) in path:
#                if (node_to_node_distance[iv,v] != 0):
#                    print node_to_node_distance[iv,v],'->', pathlen - ilen
                node_to_node_distance[iv,v] = pathlen - ilen
                node_to_node_distance[v,iv] = pathlen - ilen
            npath = path[:]
            npath.append((v,pathlen))
            dfs(npath)
    for leaf in leaves:
        visited = [None] * len(adj)
        visited[leaf] = True
        path = [(leaf,0)]
        dfs(path)
    for i,m in enumerate(leaves):
        for j,n in enumerate(leaves):
            leaf_to_leaf_distance[i,j] = node_to_node_distance[m,n]
    return leaf_to_leaf_distance

m = distances_between_leaves(4,[(0, 4, 11), (1, 4, 2), (2, 5, 6), (3, 5, 7), (4, 0, 11), (4, 1, 2), (4, 5, 4), (5, 4, 4), (5, 3, 7), (5, 2, 6)])
res = np.array([[ 0, 13, 21, 22],[13,  0, 12, 13],[21, 12,  0, 13],[22, 13, 13,  0]])
assert not (m != res).sum()

def limb_length(n, j, d):
    """
    CODE CHALLENGE: Solve the Limb Length Problem.
    Input: An integer n, followed by an integer j between 0 and n, followed by a space-separated
    additive distance matrix D (whose elements are integers).
    Output: The limb length of the leaf in Tree(D) corresponding to the j-th row of this distance
    matrix (use 0-based indexing).
    """
    assert n > 2
    assert d.shape == (n,n)
    i = (j+1) % n
    k = (j+2) %n
    minlen = (d[i,j] + d[j,k] - d[i,k])/2
    for i in range(n):
        for k in range(i+1,n):
            if i==j or k==j:
                continue
            currlen = (d[i,j] + d[j,k] - d[i,k])/2
            minlen = min(minlen,currlen)
    return minlen

def limb_length_linear(n, j, d):
    """
    CODE CHALLENGE: Solve the Limb Length Problem in linear time.
    Input: An integer n, followed by an integer j between 0 and n, followed by a space-separated
    additive distance matrix D (whose elements are integers).
    Output: The limb length of the leaf in Tree(D) corresponding to the j-th row of this distance
    matrix (use 0-based indexing).
    """
    assert n > 2
    assert d.shape == (n,n)
    i = (j+1) % n
    k = (j+2) %n
    minlen = (d[i,j] + d[j,k] - d[i,k])/2
    for k in range(n):
        if i==j or k==j:
            continue
        currlen = (d[i,j] + d[j,k] - d[i,k])/2
        minlen = min(minlen,currlen)
    return minlen

assert map(lambda j:limb_length(4,j,np.array([[0,13,21,22],[13,0,12,13],[21,12,0,13],[22,13,13,0]])), range(4)) == [11,2,6,7]
assert map(lambda j:limb_length_linear(4,j,np.array([[0,13,21,22],[13,0,12,13],[21,12,0,13],[22,13,13,0]])), range(4)) == [11,2,6,7]


def additive_phylogeny(n,d):
    """
    CODE CHALLENGE: Implement AdditivePhylogeny to solve the Distance-Based Phylogeny Problem.
    Input: An integer n followed by a space-separated n x n distance matrix.
    Output: A weighted adjacency list for the simple tree fitting this matrix.
    """
    def find_path(i,j):
        visited = [None] * (max(adj.keys())+1)
        def dfs(path):
            for (v,w) in adj[path[-1][0]]:
                if visited[v] == True:
                    continue
                visited[v] = True
                pathlen = path[-1][1] + w
                npath = path[:]                
                npath.append((v,pathlen))
                if (v == j):
                    # found the node that ends path.
                    return npath
                result = dfs(npath)
                if result is not None:
                    return result
            return
        return dfs([(i,0)])
        
    def add_leaf(leaf,i,j,x,w):
        # add leaf on the path from i to j,
        # attached at x distance from i
        # with weigth w
        path = find_path(i,j)
        # check if an x distant parent node already exists in path p
        parent = None
        for k in range(len(path)-1):
            if path[k][1] == x:
                # parent node already exists
                parent = path[k][0]
                break
            if path[k][1] < x and x < path[k+1][1]:
                # cut the u->v:w0 edge into
                # u->parent:w1 , parent->v:w2
                u = path[k][0]
                v = path[k+1][0]
                w0 = path[k+1][1] - path[k][1]
                w1 = x - path[k][1]
                w2 = w0 - w1
                # find a new node slot
                parent = max(adj.keys()+[n-1])+1 

                adj[u].remove((v,w0))
                adj[v].remove((u,w0))
                adj[parent] = []
                
                adj[u].append((parent,w1))
                adj[parent].append((u,w1))

                adj[v].append((parent,w2))
                adj[parent].append((v,w2))                

                break
        assert parent is not None

        adj.setdefault(leaf,[]).append((parent,w))
        adj[parent].append((leaf,w))        
        return
    
    def find_condition(d,l):
        # (i,l,k) ← three leaves such that
        # Di,k = Di,l + Dl,k
        # i < k < l
        for i in range(l):
            for k in range(i+1,l):
                if d[i,k] == d[i,l]+d[l,k]:
                    return (i,k)
        assert 0
        return
    
    def recursive(nn):
        if nn==2:
            return
        limb = limb_length_linear(nn,nn-1,d[:nn,:nn])
        for j in range(nn-1):
            d[nn-1, j] -= limb
            d[j, nn-1] -= limb
        (i, k) = find_condition(d,nn-1)
        x = d[i,nn-1]
        recursive(nn-1)
        add_leaf(nn-1,i,k,x,limb)
        return
        
    adj = {0: [(1,d[0,1]),],1: [(0,d[1,0]),]}
    recursive(n)
    return adj
   
d = np.array([[0,13,21,22],[13,0,12,13],[21,12,0,13],[22,13,13,0]])
T = additive_phylogeny(4,d)
assert T == {0: [(4, 11)], 1: [(4, 2)], 2: [(5, 6)], 3: [(5, 7)], 4: [(0, 11), (1, 2), (5, 4)], 5: [(4, 4), (2, 6), (3, 7)]}


############################################################
#fpath = 'C:/Users/ngaude/Downloads/'
#fpath = '/home/ngaude/Downloads/'
fpath = 'C:/Users/Utilisateur/Downloads/'
############################################################

#fname = fpath + 'dataset_10328_11.txt'
#with open(fname, "r") as f:
#    lines = f.read().strip().split('\n')
#    n = int(lines[0])
#    ladj = [ ( int(l.split('->')[0]), int(l.split('->')[1].split(':')[0]), int(l.split('->')[1].split(':')[1])) for l in lines[1:] ]
#    m = distances_between_leaves(n,ladj)
#with open(fname+'.out', "w") as f:
#   for row in m:
#       f.write(' '.join(map(str,row)))
#       f.write('\n')

#fname = fpath + 'dataset_10329_11.txt'
#with open(fname, "r") as f:
#    lines = f.read().strip().split('\n')
#    n = int(lines[0])
#    j = int(lines[1])
#    d = map(lambda r: map(int, r.split(' ')), lines[2:])
#    d = np.array(d)
#print limb_length_linear(n,j,d)

#fname = fpath + 'dataset_10330_6.txt'
#with open(fname, "r") as f:
#    lines = f.read().strip().split('\n')
#    n = int(lines[0])
#    d = map(lambda r: map(int, r.split(' ')), lines[1:])
#    d = np.array(d)
#T = additive_phylogeny(n,d)
#with open(fname+'.out', "w") as f:
#   for u,vw in T.iteritems():
#       for (v,w) in vw:
#           f.write(str(u)+'->'+str(v)+':'+str(w)+'\n')

############################################################
# QUIZZ
############################################################


D = np.array([[0,13,16,10],[13,0,21,15],[16,21,0,18],[10,15,18,0]])
print 'limb_length',limb_length(4, 2, D)

