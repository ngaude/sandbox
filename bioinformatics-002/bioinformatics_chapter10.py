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



#s = '4\n0->4:11\n1->4:2\n2->5:6\n3->5:7\n4->0:11\n4->1:2\n4->5:4\n5->4:4\n5->3:7\n5->2:6'
#lines = s.split('\n')
#n = int(lines[0])
#ladj = [ ( int(l.split('->')[0]), int(l.split('->')[1].split(':')[0]), int(l.split('->')[1].split(':')[1])) for l in lines[1:] ]
#m = distances_between_leaves(n,ladj)

fname = 'C:/Users/ngaude/Downloads/dataset_10328_11.txt'
with open(fname, "r") as f:
    lines = f.read().strip().split('\n')
    n = int(lines[0])
    ladj = [ ( int(l.split('->')[0]), int(l.split('->')[1].split(':')[0]), int(l.split('->')[1].split(':')[1])) for l in lines[1:] ]
    m = distances_between_leaves(n,ladj)
with open(fname+'.out', "w") as f:
   for row in m:
       f.write(' '.join(map(str,row)))
       f.write('\n')

    
