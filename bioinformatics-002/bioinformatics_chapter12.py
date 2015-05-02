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
    m = n+1
    for uedge in uedges:
        node = uedge[0]
        child = uedge[1]
        m = max(node+1,child+1,m)
        utree.setdefault(node,[]).append(child)
    
    # m, total node count
    
     # given edge create a rooted tree from unrooted tree
    def get_rooted_tree(uedge):
        (lnode,rnode) = uedge
        root = m
        visited = [False]*(m+1)
        edges = []
        rooted_tree = copy.deepcopy(utree)
        rooted_tree[rnode].remove(lnode)
        rooted_tree[lnode].remove(rnode)
        rooted_tree[root] = [lnode,rnode]
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
        

def edge2tree(edges):
    tree = {}
    for e in edges:
        node = e[0]
        child = e[1]
        tree.setdefault(node,[]).append(child)
    return tree
    
def tree2edge(tree):
    edges = []
    for node,children in tree.iteritems():
        for child in children:
            edges.append((node,child))
    return edges

def tree_nearest_neighbors(e,utree):
    '''
    CODE CHALLENGE: Solve the Nearest Neighbors of a Tree Problem.
    Input: Two internal nodes a and b specifying an edge e, followed by an adjacency
    list of an unrooted binary tree.
    Output: Two adjacency lists representing the nearest neighbors of the tree with
    respect to e. Separate the adjacency lists with a blank line.
    '''
    a = e[0]
    b = e[1]
    
    atree = utree[a][:]
    atree.remove(b)
    w = atree[0]
    x = atree[1]
    btree = utree[b][:]
    btree.remove(a)
    y = btree[0]
    z = btree[1] 

#    # neighbor utree1 is like wya <=>bxz :
#    utree1 = copy.copy(utree)
#    utree1[a] = [b,y,w]
#    utree1[y] = utree1[y][:]
#    utree1[y].remove(b)
#    utree1[y].append(a)
#    utree1[b] = [a,x,z]
#    utree1[x] = utree1[x][:]
#    utree1[x].remove(a)
#    utree1[x].append(b)
#        
#    # neighbor utree2 is like wza <=>bxy :
#    utree2 = copy.copy(utree)
#    utree2[a] = [b,z,w]
#    utree2[z] = utree2[z][:]
#    utree2[z].remove(b)
#    utree2[z].append(a)
#    utree2[b] = [a,x,y]
#    utree2[x] = utree2[x][:]
#    utree2[x].remove(a)
#    utree2[x].append(b)
#    return (utree1,utree2)
    
    # neighbor utree1 is like wya <=>bxz :
    utree1 = copy.deepcopy(utree)
    utree1[a] = [b,y,w]
    utree1[y].remove(b)
    utree1[y].append(a)
    utree1[b] = [a,x,z]
    utree1[x].remove(a)
    utree1[x].append(b)
        
    # neighbor utree2 is like wza <=>bxy :
    utree2 = copy.deepcopy(utree)
    utree2[a] = [b,z,w]
    utree2[z].remove(b)
    utree2[z].append(a)
    utree2[b] = [a,x,y]
    utree2[x].remove(a)
    utree2[x].append(b)
    return (utree1,utree2)

#e = [(4,0),(0,4),(4,1),(1,4),(5,2),(5,3),(2,5),(3,5),(4,5),(5,4)]
#tree = {0: [4], 1: [4], 2: [5], 3: [5], 4: [0, 1, 5], 5: [2, 3, 4]}
#assert tree == edge2tree(e)
#assert sorted(e) == sorted(tree2edge(tree))
#res1 = [(0, 4), (1, 5), (2, 4), (3, 5), (4, 0), (4, 2), (4, 5), (5, 1), (5, 3), (5, 4)]
#res2 = [(0, 4), (1, 5), (2, 5), (3, 4), (4, 0), (4, 3), (4, 5), (5, 1), (5, 2), (5, 4)]
#edge1,edge2 = map(tree2edge, tree_nearest_neighbors((4,5), edge2tree(e)))
#assert res1 == sorted(edge1)
#assert res2 == sorted(edge2)

def large_parsimony_problem(n,edges,labels):
    '''
    CODE CHALLENGE: Implement the nearest neighbor interchange heuristic for the Large Parsimony Problem.
    Input: An integer n, followed by an adjacency list for an unrooted binary tree whose n leaves are
    labeled by DNA strings and whose internal nodes are labeled by integers.
    Output: The parsimony score and unrooted labeled tree obtained after every step of the nearest
    neighbor interchange heuristic. Each step should be separated by a blank line.
    '''
    print '>>> go: !'
    (p,e) = small_parsimony_unrooted_problem(n,edges,labels)
    ret = [(p,e),]
    parsimony = p
    while True:
        minimum_achieved = True
        print 'parsimony',parsimony
        candidate_edges = set([(min(a,b),max(a,b)) for (a,b) in edges if a>=n and b>=n])
        print candidate_edges
        for e in candidate_edges:
            print e,len(candidate_edges)
            if e[0] < n or e[1] < n:
                # simply skip leaves
                continue
            edges1,edges2 = map(tree2edge, tree_nearest_neighbors(e, edge2tree(edges)))
            (p,e) = small_parsimony_unrooted_problem(n,edges1,labels)
            if p < parsimony:
                parsimony = p
                nedges = edges1
                minimum_achieved = False
            (p,e) = small_parsimony_unrooted_problem(n,edges2,labels)
            if p < parsimony:
                parsimony = p
                nedges = edges2
                minimum_achieved = False
        if minimum_achieved == True:
            break
        else:
            edges = nedges
            (p,e) = small_parsimony_unrooted_problem(n,edges,labels)
            ret.append((p,e))
            
    return ret 
        
#n = 4
#labels = ['CGAAGATTCTAA','ATGCCGGGCTCG','CTTTTAGAAGCG','AACTCATGATAT']
#v = [(0,4),(1,4),(2,5),(3,5),(5,3),(5,2),(5,4),(4,1),(4,0),(4,5)]
#ret = large_parsimony_problem(n,v,labels)
#assert ret[-1][0] == 21
            
def parsing_large_parsimony_problem_input(lines):
    n = int(lines[0])
    labels = []
    edges = []
    for l in lines[1:]:
        a = l.split('->')[0]
        b = l.split('->')[1]
        if a.isdigit() == True:
            a = int(a)
        else:
            if a not in labels:
                labels.append(a)
            a = labels.index(a)
        if b.isdigit() == True:
            b = int(b)
        else:
            if b not in labels:
                labels.append(b)
            b = labels.index(b)
        edges.append((a,b))
    return n,edges,labels


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
#fname = fpath + 'dataset_10335_12.txt'
#with open(fname, "r") as f:
#    lines = f.read().strip().split('\n')
#    n = int(lines[0])
#    labels = map(lambda l:l.split('->')[0],lines[1:2*n+1:2])
#    v  = map(lambda (i,l):(i,int(l.split('->')[1])),enumerate(lines[1:2*n+1:2]))
#    v += map(lambda (i,l):(int(l.split('->')[0]),i),enumerate(lines[2:2*n+1:2]))
#    v += map(lambda l:(int(l.split('->')[0]),int(l.split('->')[1])),lines[2*n+1:])
#    (p,e) = small_parsimony_unrooted_problem(n,v,labels)
#with open(fname+'.out', "w") as f:
#    f.write(str(p)+'\n')
#    for (a,b) in e:
#        f.write(a+'->'+b+':'+str(hamming(a,b))+'\n')
#        f.write(b+'->'+a+':'+str(hamming(a,b))+'\n')

#fname = fpath + 'dataset_10336_6.txt'
#with open(fname, "r") as f:
#    lines = f.read().strip().split('\n')
#    a = int(lines[0].split(' ')[0])
#    b = int(lines[0].split(' ')[1])
#    v = map(lambda l:(int(l.split('->')[0]),int(l.split('->')[1])),lines[1:])
#edge1,edge2 = map(tree2edge, tree_nearest_neighbors((a,b), edge2tree(v)))
#s = '\n'.join(map(lambda e: str(e[0])+'->'+str(e[1]),edge1))
#s += '\n'+'\n'
#s += '\n'.join(map(lambda e: str(e[0])+'->'+str(e[1]),edge2))
#with open(fname+'.out', "w") as f:
#    f.write(s)

fname = fpath + 'dataset_10336_8.txt'
#fname = fpath + 'Large_Parsimony_Heuristic_with_NNI.txt'
with open(fname, "r") as f:
    lines = f.read().strip().split('\n')
    n,edges,labels = parsing_large_parsimony_problem_input(lines)
pes = large_parsimony_problem(n,edges,labels)
with open(fname+'.out', "w") as f:
    for p,e in pes:
        f.write(str(p)+'\n')
        for (a,b) in e:
            f.write(a+'->'+b+':'+str(hamming(a,b))+'\n')
            f.write(b+'->'+a+':'+str(hamming(a,b))+'\n')