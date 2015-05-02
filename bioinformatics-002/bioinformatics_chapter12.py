# -*- coding: utf-8 -*-
"""
Created on Sat May 02 00:53:05 2015

@author: ngaude
"""

import numpy as np
import copy

def hamming(s1,s2):
    assert len(s1) == len(s2)
    d = 0
    for i in range(len(s1)):
        if s1[i]!=s2[i]:
            d+=1
    return d

def small_parsimony_problem(n,edges,labels):
    """
    Small Parsimony Problem. Find the most parsimonious labeling of the internal nodes of a rooted tree. 
    Input: A rooted binary tree with each leaf labeled by a string of length m.
    Output: A labeling of all other nodes of the tree by strings of length m 
    that minimizes the parsimony score of the tree.

    CODE CHALLENGE: Implement SmallParsimony to solve the Small Parsimony Problem.
    Input: An integer n followed by an adjacency list for a rooted binary tree with n leaves
    labeled by DNA strings.
    Output: The minimum parsimony score of this tree, followed by the adjacency list of the
    tree corresponding to labeling internal nodes by DNA strings in order to minimize the
    parsimony score of the tree.
    """


    # m, total node count
    m = 2*n-1    
    
    # compute tree and reverse-tree from edges
    tree = {}
    parent = {}
    for edge in edges:
        node = edge[0]
        child = edge[1]
        tree.setdefault(node,[]).append(child)
        parent[child] = node

    # get the root from parent list
    root = parent[parent.keys()[0]]
    while root in parent.keys():
        root = parent[root]

    # l, size of string labels
    l = len(labels[0])
    
    # alphabet of characters of labels
    alphabet = sorted(list(set(''.join(labels))))
    
    # k, number of characters in string labels
    k = len(alphabet)
    
    # d, dictionnary of every character position in the alphabet
    d = dict(zip(alphabet,range(k)))

    # string labels 
    s = [[' ']*l for i in range(m)]
    
    sk = np.ndarray(shape=(m,k,l), dtype=int)
    
    # maximum parsimony value is (m-1)*l
    sk.fill((m-1)*l)
    
    # fill leaf sk according to labels
    for i,label in enumerate(labels):
        s[i] = list(label)
        for j,c in enumerate(label):
            sk[i,d[c],j] = 0
    
    # depth_first search for each string element to fill sk values
    for i in range(l):
        def dfs_sk(node):
            if node < n:
                # leaf is at tree botton, simply return
                return
            lnode = tree[node][0]
            rnode = tree[node][1]
            dfs_sk(lnode)
            dfs_sk(rnode)
            for j in range(k):
                mask = np.ones(k)
                mask[j] = 0
                sk[node,j,i] = min(sk[lnode,:,i]+mask) + min(sk[rnode,:,i]+mask)
            return
        dfs_sk(root)
        
    parsimony = sum(sk[root].min(axis=0))
    
    # depth_first search to back propagate the internal node string s values 
    for i in range(l):        
        def dfs_s(node):
            
            if node < n:
                # leaf is at tree botton, simply return
                return
            c = sk[node,:,i]
            if node == root:
                # when root simply choose the min score ever                
                s[node][i] = alphabet[c.argmin()]
            else:
                pnode = parent[node]
                j = d[s[pnode][i]]
                mask = np.ones(k)
                mask[j] = 0
                c += mask
                s[node][i] = alphabet[c.argmin()]
            
            lnode = tree[node][0]
            rnode = tree[node][1]
            dfs_s(lnode)
            dfs_s(rnode)
        dfs_s(root)
    
    ret = []
    for node,(lnode,rnode) in tree.iteritems():
        ps = ''.join(s[node])
        ls = ''.join(s[lnode])
        rs = ''.join(s[rnode])
        ret.append((ps,ls))
        ret.append((ps,rs))
    lbranch = (''.join(s[root]), ''.join(s[tree[node][0]]))
    rbranch = (''.join(s[root]), ''.join(s[tree[node][1]]))
    return (parsimony,ret[:],rbranch,lbranch)
        

n = 4
v = [(4,0),(4,1),(5,2),(5,3),(6,4),(6,5)]
labels = ['CAAATCCC', 'ATTGCGAC', 'CTGCGCTG', 'ATGGACGA']
(p,e,r,l) = small_parsimony_problem(n,v,labels)
assert p == 16

#n = 3
#v = [(3,0),(3,4),(4,1),(4,2)]
#labels = ['AAA', 'ATA', 'AGA']
#(p,e,r,l) = small_parsimony_problem(n,v,labels)
#print p,e,r,l

def small_parsimony_unrooted_problem(n,uedges,labels):
    '''
    CODE CHALLENGE: Solve the Small Parsimony in an Unrooted Tree Problem.
    Input: An integer n followed by an adjacency list for an unrooted binary tree with n leaves
    labeled by DNA strings.
    Output: The minimum parsimony score of this tree, followed by the adjacency list of the
    tree corresponding to labeling internal nodes by DNA strings in order to minimize the
    parsimony score of the tree.
    '''
    # compute unrooted tree from edges
    utree = {}
    for uedge in uedges:
        node = uedge[0]
        child = uedge[1]
        utree.setdefault(node,[]).append(child)
    
    # m, total node count
    m = 2*n-1 
    
     # given edge create a rooted tree from unrooted tree
    def get_rooted_tree(uedge):
        (lnode,rnode) = uedge
        root = m-1
        visited = [False]*m
        edges = []
        rooted_tree = copy.deepcopy(utree)
        rooted_tree[rnode].remove(lnode)
        rooted_tree[lnode].remove(rnode)
        rooted_tree[root] = (lnode,rnode)
        def dfs_rooted_tree(node):
            if (node < n) or (visited[node] == True):
                # tree bottom or already visited, simply return
                return
            visited[node] = True
            for child in rooted_tree[node]:
                if (visited[child] == False ):
                    edges.append((node,child))
                    dfs_rooted_tree(child)
            return
        dfs_rooted_tree(root)      
        return edges

    perls = []
    for uedge in set([(min(a,b),max(a,b)) for (a,b) in uedges]):
        edges = get_rooted_tree(uedge)
        perl = small_parsimony_problem(n,edges,labels)
        perls.append(perl)
    (p,e,r,l) = min(perls)
    # find root's children
    ret = [(edge[0],edge[1]) for edge in e if (edge != r and edge !=l)]
    ret.append((l[1],r[1]))
    return (p,ret)
    
n = 4
u = [(4,0),(0,4),(4,1),(1,4),(5,2),(5,3),(2,5),(3,5),(4,5),(5,4)]
labels = [ 'TCGGCCAA','CCTGGCTG','CACAGGAT','TGAGTACC']
(p,e) = small_parsimony_unrooted_problem(n,u,labels)           
assert p == 17
        
            
    

############################################################
fpath = 'C:/Users/ngaude/Downloads/'
#fpath = '/home/ngaude/Downloads/'
#fpath = 'C:/Users/Utilisateur/Downloads/'
############################################################

#fname = fpath + 'dataset_10335_10.txt'
#fname = fpath + 'Small_Parsimony.txt'
#with open(fname, "r") as f:
#    lines = f.read().strip().split('\n')
#    n = int(lines[0])
#    labels = map(lambda l:l.split('->')[1],lines[1:n+1])
#    v = map(lambda (i,l):(int(l.split('->')[0]),i),enumerate(lines[1:n+1]))
#    v += map(lambda l:(int(l.split('->')[0]),int(l.split('->')[1])),lines[n+1:])
#    (p,e,r) = small_parsimony_problem(n,v,labels)
#with open(fname+'.out', "w") as f:
#    f.write(str(p)+'\n')
#    for (a,b) in e:
#        f.write(a+'->'+b+':'+str(hamming(a,b))+'\n')
#        f.write(b+'->'+a+':'+str(hamming(a,b))+'\n')

#fname = fpath + 'Small_Parsimony_Unrooted_Tree.txt'
fname = fpath + 'dataset_10335_12.txt'
with open(fname, "r") as f:
    lines = f.read().strip().split('\n')
    n = int(lines[0])
    labels = map(lambda l:l.split('->')[0],lines[1:2*n+1:2])
    v  = map(lambda (i,l):(i,int(l.split('->')[1])),enumerate(lines[1:2*n+1:2]))
    v += map(lambda (i,l):(int(l.split('->')[0]),i),enumerate(lines[2:2*n+1:2]))
    v += map(lambda l:(int(l.split('->')[0]),int(l.split('->')[1])),lines[2*n+1:])
    (p,e) = small_parsimony_unrooted_problem(n,v,labels)
with open(fname+'.out', "w") as f:
    f.write(str(p)+'\n')
    for (a,b) in e:
        print a,b
        f.write(a+'->'+b+':'+str(hamming(a,b))+'\n')
        f.write(b+'->'+a+':'+str(hamming(a,b))+'\n')
