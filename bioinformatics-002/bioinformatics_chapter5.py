# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 20:08:03 2014

@author: ngaude
"""

import numpy as np
import copy

def dynamic_programming_change(money, coins):
    '''
    CODE CHALLENGE: Solve the Change Problem.
    Input: An integer money and an array coins = (coin1, ..., coind).
    Output: The minimum number of coins with denominations coins that changes money.
    '''
    min_num_coins = []
    min_num_coins.append(0)
    for m in range (1,money+1):
        min_num_coins.append(float("inf"))
        for i in coins:
            if (m >= i):
                if min_num_coins[m - i] +  1 < min_num_coins[m]:
                    min_num_coins[m] = min_num_coins[m - i] +  1
    return min_num_coins[money]

assert dynamic_programming_change(8074, (24,13,12,7,5,3,1)) == 338

#m = 17328
#coins = (24,15,5,3,1)
#print dynamic_programming_change(m, coins)

def manhattan_tourist(n, m, down, right):
    '''
    Find the length of a longest path in the Manhattan Tourist Problem.
    Input: Integers n and m, followed by an n × (m + 1) matrix Down 
    and an (n + 1) × m matrix Right.
    Output: The length of a longest path from source (0, 0) to sink 
    (n, m) in the n × m rectangular grid whose edges are defined by the 
    matrices Down and Right.
    Warning : n rows, m columns
    '''
    down = np.array(down).reshape(n, m+1)
    right = np.array(right).reshape(n+1, m)
    s = np.empty(shape = (n+1,m+1), dtype = int) 
    s[0,0] = 0
    for i in range(n):
        s[i+1,0] = s[i,0] + down[i,0]
    for j in range(m):
        s[0,j+1] = s[0,j] + right[0,j]
    for i in range(n):
        for j in range(m):
            s[i+1,j+1] = max(s[i,j+1]+down[i,j+1], s[i+1,j]+right[i+1,j])
#            print 's[',i+1,j+1,']=max',s[i,j+1],'+',down[i,j+1],',',s[i+1,j],'+',right[i+1,j]
    return s[n,m]


assert manhattan_tourist(4,4,((1,0,2,4,3),(4,6,5,2,1),(4,4,5,2,1),(5,6,8,5,3)),((3,2,4,0),(3,2,4,2),(0,7,3,3),(3,3,0,2),(1,3,2,2))) == 34

#fname = 'C:/Users/ngaude/Downloads/dataset_261_9.txt'
#lines = list(l.strip() for l in open(fname))
#n = int(lines[0].split(' ')[0])
#m = int(lines[0].split(' ')[1])
#down = map(lambda l : map(int, l.split(' ')), lines[1:1+n])
#assert lines[1+n] == '-'
#right = map(lambda l : map(int, l.split(' ')), lines[2+n:2+n+n+1])
#print manhattan_tourist(n, m, down, right)

def longest_common_subsequence(v, w, indel = 0, scoring = None, verbose = False, local = False):
    '''
    CODE CHALLENGE: solve the Longest Common Subsequence Problem.
    Input: Two strings v and w
    Output: A longest common subsequence of v and w. 
    (Note: more than one solution may exist, in which case you may output any one.)
    '''
    n = len(v)
    m = len(w)
    s = np.zeros(shape = (n+1,m+1), dtype = np.float)
    for i in range(n+1):
        s[i, 0] = -indel * i
    for j in range(m+1):
        s[0, j] = -indel * j
    backtrack = np.chararray(shape = (n,m))
    for i in range(n):
        for j in range(m):
            if (scoring is None):
                score = (1 if v[i] == w[j] else 0)
            else:
                score = scoring[ v[i] ][ w[j] ]
            s[i+1, j+1] = max(s[i, j+1] - indel, s[i+1, j] - indel,
                s[i, j] + score)
            if s[i+1, j+1] == s[i, j+1] - indel:
                    backtrack[i, j] = '|'
            elif s[i+1, j+1] == s[i+1, j] - indel:
                    backtrack[i, j] = '-'
            elif s[i+1, j+1] == s[i, j] + score:
                    backtrack[i, j] = '/' if v[i] == w[j] else '*'
            # if local alignement is requested,
            # allow free-taxi ride from the source e.g. min s = 0
            # thus ensure s[i+1, j+1] > 0
            if (local and s[i+1, j+1] < 0):
                s[i+1, j+1] = 0
                backtrack[i, j] = 'x'
    lcs = []
    valign = []
    walign = []
    
    if local:
        # if local alignement is requested,
        # allow free-taxi ride to the sink
        # thus ensure s[i, j] is maximum over the entire graph
        (i, j) = np.unravel_index(s.argmax(),s.shape)
    else:
        (i, j) = (n, m)

    smax = s[i, j]
    
    i -= 1
    j -= 1

#    print '------------'
#    print s
#    print backtrack
#    print local,i,j
#    print '------------'
    
    while (i >= 0 and j >= 0):
#        assert backtrack[i,j] != '*'
        if backtrack[i, j] == '|':
            walign.append('-')
            valign.append(v[i])
            i -= 1
        elif backtrack[i, j] == '-':
            walign.append(w[j])
            valign.append('-')
            j -= 1
        elif backtrack[i, j] == 'x':
            # break when a free-taxi ride back to source is allowed
            break            
        else:
            lcs.append(w[j])
            walign.append(w[j])
            valign.append(v[i])
            j -= 1
            i -= 1
    if not local:
        while (i>=0):
            walign.append('-')
            valign.append(v[i])
            i -= 1
        while (j>=0):
            valign.append('-')
            walign.append(w[j])
            j -= 1
        
    lcs.reverse()
    valign.reverse()
    walign.reverse()
    lcs = ''.join(lcs)
    valign = ''.join(valign)
    walign = ''.join(walign)
    if verbose:
        return (smax,valign,walign)
    else:
        return lcs

assert len(longest_common_subsequence('AACCTTGG','ACACTGTGA')) == 6

#v = 'ATAGTCTGCGTCCACCAAGAACTCGCGATACGGCTTGGGTGAGAGAAGAAATTGTTGGACTGTCGTGGGTATAACATCCGGGTCGAGGCGAAGGTCCGCTAGTTATCCCCTGGCGTGGCTAAATAGCAATTTTGCAGAATGCCTTATCTATCTTCAGCCATGCGCTTAACTAAATATTTTCAGTATAGTGTTACTAACGAGGAAGTCAAGTGCGTCTAATTTATTGAATAGAACTACTAGTTTTCCCCATAACGCCTAACTTTTAGAATACGATCTCTTTAGGATTAGAGATTTGCCAGGGCACACCGGGGTTGAGAGGGAAGGGGTTTGTACCTCGTGCGGCCTCAGGGGGGCGGTGTTGTCAGAATCTATCCGCACCCGCAGACCTATATTCCTACGTGGCAAGCCTGCATTTGTTAGCTGGTCCGCCTCGCGCAGGCGTCCGTCGGACCAGTGTCGAGAGCCCAATGCATTAGCAGTTCTTTGCCGTTATGGTTCTAAATGAGAAAGCACGTAGCCTTCGAATCAGTTTCCCGATTACTGGCTCCTCGACATAAAAAATCTGTCTCGAGTAGCATTCTGACGATAGGTTAGGGTGCCTGGAGTGAATTTAGCTGCATAATACCTTTGCGACCGTAGAAGACCCGGCCTGGATAGTGATCACCGCTGCTCCCCCATCTGGTTTCCTAGCTAATTGGGTCAGTGACTCTCCTCTCACTTGGGTTCTTCTTCTACGGAATCTGCGAAACACTGCAATATGGACCACAGAGCTTAAAGTATAGAGTCGCGCCAGTCACCACTCTGGTGATCAGTGAAGGGAATGATCACTCCAAACTGAACTCGCCGGTTTTAGTCTTGGCGGCCTTGTACGCTCACGGGTGAAGTAAGTACTAG'
#w = 'TTTCGCCGGGGCCTACGACAACTACACCACTTTTATTTCTCCTTATTGAATCTATGTTATCTGCGATAACGAGCAAGGGCACGGTCCGGTAGAATAAACTGGATGCATTAGGCACCATATTGAATTCAGGTGGGTACCGGTGTCCCCTCGTCCGGCGGACTAGCGGATGGCGGACATCAAGGGGGCGCACTGCCCAAGGGAAACACTCCGGGCATGCACCTCGATCACAGGTGTCTCACTCGTCACAGGTCTGCAAGTACCACGACAATTTCCTTCTCCTGGTGGATTCTTAGGGAGGAGTCCACTATACATACTCGTATCAGACTACTCAGGCCACCTCCTTAGGCGAAGGATATTCACCGGTGTGCGTTTGCAGACCCGACTGACCGTCTATGCCAGCCCTCAGTTATGCAAAATTTTACGATTTCACTGTCTACGTGGCCGGACTGGCCTGAACGGCAAAAAGGCAAGAGGATGAGCGCATATATGGGGTCCAGTTACATCACCAAATTCTAGGGCATTACTAAGCTATGACTGGCTACAGGTGTTGCGGTACCTCCCAGTAGATATTTGAGCGGAAACCGTATGACTGCAAGTTCGGATTAAGAGCCGCCGACGCTATCAGGCGCAGCGACAAATCGGAACGCGTCAGACGAACCAGGAAGAACGGCTGCTAGCCTTTACAACCTGATCAGCCCGTTGAGCGAAATGAATTACGGCTCGGGGGCAACTCTGACAGACCTCAGGTTTAAGCCTAAAACGTAGGACCCTTGAGCTTTCGCGCTATCTCCTAGAAGAAGGGAACCAGTGTCGCCTGTACGTTGCTGTAACCGTTTTTTGTACACCACAAGTCAGCCTCGCCCTCATTTTACTATGTCGGTGGCT'
#print longest_common_subsequence(v,w)


class edge_weighted_graph:
    def __init__(self):
        # set of vertex
        self.vertex = set()
        # adjacency list of set of edge : 
        # key = vertex initial
        # value = ( ..., (vertex final, distance), ...)
        self.edge = {}
        return
        
    def add_edge(self, u, v, w):
        '''
        add a weigthed edge from vertex u to v
        '''
        assert u != v
        self.vertex.add(u)
        self.vertex.add(v)
        self.edge.setdefault(u,set()).add((v,w))
    
    def remove_edge(self, u, v, w):
        '''
        remove a weighted edge from vertex u to v
        '''
        assert u != v
        d = self.edge.get(u, set())
        d.remove((v, w))
    
    def sub_graph(self, u):
        '''
        return the sub graph starting from vertex u
        '''
        sg = edge_weighted_graph()
        on_stack = []
        on_stack.append(u)
        while (on_stack):
            u = on_stack.pop()
            for (v,w) in self.edge.get(u, set()):
                sg.add_edge(u, v, w)
                on_stack.append(v)
        return sg
    
    def reverse(self):
        '''
        return the reversed graph
        '''
        r = edge_weighted_graph()
        for u,d in self.edge.iteritems():
            for (v,w) in d:
                r.add_edge(v, u, w)
        return r
    
    def sort(self):
        '''
        return a topologically sorted list of vertex
        '''
        s = []
        g = copy.deepcopy(self)
        r = g.reverse()
        candidates = [ u for u in self.vertex if not r.edge.get(u, set())]
        while candidates:
            u = candidates.pop()
#            print '---------'
#            print 'u=',u
#            print 'g.edge',g.edge
#            print 'r.edge',r.edge
            s.append(u)
            for (v,w) in set(g.edge.get(u, set())):
                # remove u->v edge from g and u-> from r
                g.remove_edge(u, v, w)
                r.remove_edge(v, u, w)
                if not r.edge[v]:
                    r.edge.pop(v, None)
                    candidates.append(v)
            g.edge.pop(u, None)
        assert not g.edge
        assert not r.edge
        return s
    
    def __str__(self):
        s = ''
        for u,d in self.edge.iteritems():
            for (v,w) in d:
                 s += str(u) + '->' + str(v) + ':' + str(w) + '\n'
        return s

   
def dag_longest_path(source, sink, edges):
    '''
    Input: 
    the source node of a graph, 
    the sink node of the graph,
    followed by a list of edges in the graph.    
    Output: longest path from source to sink as a the list of nodes.
    '''
    g = edge_weighted_graph()
    for (u,v,w) in edges:
        g.add_edge(u, v, w)
    g = g.sub_graph(source)
    order = g.sort()
    assert source in order
    assert sink in order
    a = order.index(source)
    b = order.index(sink)
    assert a < b
    assert sink in order
    # order contains the topological order of graph
    order = order[a+1:b+1]
    # path from-to source is of len 0
    s = {source:0}
    bt = {source:None}
    # get the incoming the edges
    r = g.reverse()
    # compute the max path len along the topological order 
    # from source and until sink is met
    for u in order:
        lmax = float('-Inf')
        for (v,w) in r.edge[u]:            
            l = s[v]+w
            if (l > lmax):
                lmax = l
                s[u] = l
                bt[u] = v
    # backtrack the longest path
    u = order[-1]
    path = []
    while (not u is None):
        path.append(u)
        u = bt[u] 
    path.reverse()
    return s[sink],path
    
assert dag_longest_path(0,4,((0,1,7),(0,2,4),(2,3,2),(1,4,1),(3,4,3))) == (9, [0, 2, 3, 4])

#fname = 'C:/Users/ngaude/Downloads/dataset_245_7.txt'
#lines = list(l.strip() for l in open(fname))
#source = int(lines[0])
#sink = int(lines[1])
#def parse(l):
#    a = l.split('->')
#    b = a[1].split(':')
#    return int(a[0]), int(b[0]), int(b[1])
#edges = map(parse, lines[2:])
#l,p = dag_longest_path(source, sink, edges)
#print l
#print '->'.join(map(str,p))


blosum62 = {'A': {'A': 4, 'C': 0, 'E': -1, 'D': -2, 'G': 0, 'F': -2, 'I': -1, 'H': -2, 'K': -1, 'M': -1, 'L': -1, 'N': -2, 'Q': -1, 'P': -1, 'S': 1, 'R': -1, 'T': 0, 'W': -3, 'V': 0, 'Y': -2}, 'C': {'A': 0, 'C': 9, 'E': -4, 'D': -3, 'G': -3, 'F': -2, 'I': -1, 'H': -3, 'K': -3, 'M': -1, 'L': -1, 'N': -3, 'Q': -3, 'P': -3, 'S': -1, 'R': -3, 'T': -1, 'W': -2, 'V': -1, 'Y': -2}, 'E': {'A': -1, 'C': -4, 'E': 5, 'D': 2, 'G': -2, 'F': -3, 'I': -3, 'H': 0, 'K': 1, 'M': -2, 'L': -3, 'N': 0, 'Q': 2, 'P': -1, 'S': 0, 'R': 0, 'T': -1, 'W': -3, 'V': -2, 'Y': -2}, 'D': {'A': -2, 'C': -3, 'E': 2, 'D': 6, 'G': -1, 'F': -3, 'I': -3, 'H': -1, 'K': -1, 'M': -3, 'L': -4, 'N': 1, 'Q': 0, 'P': -1, 'S': 0, 'R': -2, 'T': -1, 'W': -4, 'V': -3, 'Y': -3}, 'G': {'A': 0, 'C': -3, 'E': -2, 'D': -1, 'G': 6, 'F': -3, 'I': -4, 'H': -2, 'K': -2, 'M': -3, 'L': -4, 'N': 0, 'Q': -2, 'P': -2, 'S': 0, 'R': -2, 'T': -2, 'W': -2, 'V': -3, 'Y': -3}, 'F': {'A': -2, 'C': -2, 'E': -3, 'D': -3, 'G': -3, 'F': 6, 'I': 0, 'H': -1, 'K': -3, 'M': 0, 'L': 0, 'N': -3, 'Q': -3, 'P': -4, 'S': -2, 'R': -3, 'T': -2, 'W': 1, 'V': -1, 'Y': 3}, 'I': {'A': -1, 'C': -1, 'E': -3, 'D': -3, 'G': -4, 'F': 0, 'I': 4, 'H': -3, 'K': -3, 'M': 1, 'L': 2, 'N': -3, 'Q': -3, 'P': -3, 'S': -2, 'R': -3, 'T': -1, 'W': -3, 'V': 3, 'Y': -1}, 'H': {'A': -2, 'C': -3, 'E': 0, 'D': -1, 'G': -2, 'F': -1, 'I': -3, 'H': 8, 'K': -1, 'M': -2, 'L': -3, 'N': 1, 'Q': 0, 'P': -2, 'S': -1, 'R': 0, 'T': -2, 'W': -2, 'V': -3, 'Y': 2}, 'K': {'A': -1, 'C': -3, 'E': 1, 'D': -1, 'G': -2, 'F': -3, 'I': -3, 'H': -1, 'K': 5, 'M': -1, 'L': -2, 'N': 0, 'Q': 1, 'P': -1, 'S': 0, 'R': 2, 'T': -1, 'W': -3, 'V': -2, 'Y': -2}, 'M': {'A': -1, 'C': -1, 'E': -2, 'D': -3, 'G': -3, 'F': 0, 'I': 1, 'H': -2, 'K': -1, 'M': 5, 'L': 2, 'N': -2, 'Q': 0, 'P': -2, 'S': -1, 'R': -1, 'T': -1, 'W': -1, 'V': 1, 'Y': -1}, 'L': {'A': -1, 'C': -1, 'E': -3, 'D': -4, 'G': -4, 'F': 0, 'I': 2, 'H': -3, 'K': -2, 'M': 2, 'L': 4, 'N': -3, 'Q': -2, 'P': -3, 'S': -2, 'R': -2, 'T': -1, 'W': -2, 'V': 1, 'Y': -1}, 'N': {'A': -2, 'C': -3, 'E': 0, 'D': 1, 'G': 0, 'F': -3, 'I': -3, 'H': 1, 'K': 0, 'M': -2, 'L': -3, 'N': 6, 'Q': 0, 'P': -2, 'S': 1, 'R': 0, 'T': 0, 'W': -4, 'V': -3, 'Y': -2}, 'Q': {'A': -1, 'C': -3, 'E': 2, 'D': 0, 'G': -2, 'F': -3, 'I': -3, 'H': 0, 'K': 1, 'M': 0, 'L': -2, 'N': 0, 'Q': 5, 'P': -1, 'S': 0, 'R': 1, 'T': -1, 'W': -2, 'V': -2, 'Y': -1}, 'P': {'A': -1, 'C': -3, 'E': -1, 'D': -1, 'G': -2, 'F': -4, 'I': -3, 'H': -2, 'K': -1, 'M': -2, 'L': -3, 'N': -2, 'Q': -1, 'P': 7, 'S': -1, 'R': -2, 'T': -1, 'W': -4, 'V': -2, 'Y': -3}, 'S': {'A': 1, 'C': -1, 'E': 0, 'D': 0, 'G': 0, 'F': -2, 'I': -2, 'H': -1, 'K': 0, 'M': -1, 'L': -2, 'N': 1, 'Q': 0, 'P': -1, 'S': 4, 'R': -1, 'T': 1, 'W': -3, 'V': -2, 'Y': -2}, 'R': {'A': -1, 'C': -3, 'E': 0, 'D': -2, 'G': -2, 'F': -3, 'I': -3, 'H': 0, 'K': 2, 'M': -1, 'L': -2, 'N': 0, 'Q': 1, 'P': -2, 'S': -1, 'R': 5, 'T': -1, 'W': -3, 'V': -3, 'Y': -2}, 'T': {'A': 0, 'C': -1, 'E': -1, 'D': -1, 'G': -2, 'F': -2, 'I': -1, 'H': -2, 'K': -1, 'M': -1, 'L': -1, 'N': 0, 'Q': -1, 'P': -1, 'S': 1, 'R': -1, 'T': 5, 'W': -2, 'V': 0, 'Y': -2}, 'W': {'A': -3, 'C': -2, 'E': -3, 'D': -4, 'G': -2, 'F': 1, 'I': -3, 'H': -2, 'K': -3, 'M': -1, 'L': -2, 'N': -4, 'Q': -2, 'P': -4, 'S': -3, 'R': -3, 'T': -2, 'W': 11, 'V': -3, 'Y': 2}, 'V': {'A': 0, 'C': -1, 'E': -2, 'D': -3, 'G': -3, 'F': -1, 'I': 3, 'H': -3, 'K': -2, 'M': 1, 'L': 1, 'N': -3, 'Q': -2, 'P': -2, 'S': -2, 'R': -3, 'T': 0, 'W': -3, 'V': 4, 'Y': -1}, 'Y': {'A': -2, 'C': -2, 'E': -2, 'D': -3, 'G': -3, 'F': 3, 'I': -1, 'H': 2, 'K': -2, 'M': -1, 'L': -1, 'N': -2, 'Q': -1, 'P': -3, 'S': -2, 'R': -2, 'T': -2, 'W': 2, 'V': -1, 'Y': 7}}

assert longest_common_subsequence('PLEASANTLY','MEANLY', indel = 5, scoring = blosum62, verbose = True)[0] == 8

#v = 'ILYPRQSMICMSFCFWDMWKKDVPVVLMMFLERRQMQSVFSWLVTVKTDCGKGIYNHRKYLGLPTMTAGDWHWIKKQNDPHEWFQGRLETAWLHSTFLYWKYFECDAVKVCMDTFGLFGHCDWDQQIHTCTHENEPAIAFLDLYCRHSPMCDKLYPVWDMACQTCHFHHSWFCRNQEMWMKGDVDDWQWGYHYHTINSAQCNQWFKEICKDMGWDSVFPPRHNCQRHKKCMPALYAGIWMATDHACTFMVRLIYTENIAEWHQVYCYRSMNMFTCGNVCLRCKSWIFVKNYMMAPVVNDPMIEAFYKRCCILGKAWYDMWGICPVERKSHWEIYAKDLLSFESCCSQKKQNCYTDNWGLEYRLFFQSIQMNTDPHYCQTHVCWISAMFPIYSPFYTSGPKEFYMWLQARIDQNMHGHANHYVTSGNWDSVYTPEKRAGVFPVVVPVWYPPQMCNDYIKLTYECERFHVEGTFGCNRWDLGCRRYIIFQCPYCDTMKICYVDQWRSIKEGQFRMSGYPNHGYWFVHDDHTNEWCNQPVLAKFVRSKIVAICKKSQTVFHYAYTPGYNATWPQTNVCERMYGPHDNLLNNQQNVTFWWKMVPNCGMQILISCHNKMKWPTSHYVFMRLKCMHVLMQMEYLDHFTGPGEGDFCRNMQPYMHQDLHWEGSMRAILEYQAEHHRRAFRAELCAQYDQEIILWSGGWGVQDCGFHANYDGSLQVVSGEPCSMWCTTVMQYYADCWEKCMFA'
#w = 'ILIPRQQMGCFPFPWHFDFCFWSAHHSLVVPLNPQMQTVFQNRGLDRVTVKTDCHDHRWKWIYNLGLPTMTAGDWHFIKKHVVRANNPHQWFQGRLTTAWLHSTFLYKKTEYCLVRHSNCCHCDWDQIIHTCAFIAFLDLYQRHWPMCDKLYCHFHHSWFCRNQEMSMDWNQWFPWDSVPRANCLEEGALIALYAGIWANSMKRDMKTDHACTVRLIYVCELHAWLKYCYTSINMLCGNVCLRCKSWIFVKLFYMYAPVVNTIEANSPHYYKRCCILGQGICPVERKSHCEIYAKDLLSFESCCSQKQNCYTDNWGLEYRLFFQHIQMECTDPHANRGWTSCQTAKYWHFNLDDRPPKEFYMWLQATPTDLCMYQHCLMFKIVKQNFRKQHGHANPAASTSGNWDSVYTPEKMAYKDWYVSHPPVDMRRNGSKMVPVWYPPGIWHWKQSYKLTYECFFTVPGRFHVEGTFGCNRWDHQPGTRRDRQANHQFQCPYSDTMAIWEHAYTYVDQWRSIKEGQMPMSGYPNHGQWNVHDDHTNEQERSPICNQPVLAKFVRSKNVSNHEICKKSQTVFHWACEAQTNVCERMLNNQHVAVKRNVTFWWQMVPNCLWSCHNKMTWPTRPEQHRLFFVKMRLKCMHEYLDVAPSDFCRNMQAYMHSMRAILEYQADFDLKRRLRAIAPMDLCAQYDQEIILWSGGYIYDQSLQVVSCEGCSYYADCYVKCINVKEKCMFA'
#(a,b,c) = longest_common_subsequence(v,w, indel = 5, scoring = blosum62, verbose = True)
#print int(a)
#print b
#print c

#v = 'QTMDHMWPPYEHLYVRTKWHPFRTMRHMVARTPNNKLFIVGRIRSNAKQIFIGYLVKKAHQNDHIMPMQFDPQTSNIPYDKDGHNHLTYGHKINWLEMERGWEWWTFTDIHGLPGCHGYQICFKGAGFIFHRKRPMVVGCWEQCRSTNHRFVQCNDIEVTFYCCVFDHRFQWVMHEGPDMEMWFVCGDIRTAETGQSICDHQCPKCNQKWDQKYSWCERENERDHCPNFGAGSNLIPRGGWTLADKYWGVMFPESVEWCVCPEVYRTGQTTMMGRHTIKKKLAQTCKNHWQWDMFCKKAIRAKSQEWMIMTQEKTMEQHRLVYQTWNPVKQSKMETEQTYECGCPLQISLKFGIKNHRKYSFPIRNEYRTAGYGFRTSDLDNWPMGLVDFARIIQVEDEPLKLLFGTIHKGACQLNATFGMLECHRVSSGRPTFYCHVRMHTDFIQYTRNDNHDCNCKGEYFCDNDGFYWFAWEPMLPLSDDPLEAGYMAHQHACIFTQHRAPCYMLTYNPCSQMHENHKKLVRMANNGWAFCRAQDMTLHAPEGFDCWWRPGRQAHVYVHRLPLYRHYYIHYYCDVLKFHACMSHLMNTWVGCGLSGWDYIRHNCEFWCYMIFVFYNFHMYWPTQMIFAMKTAGKPIWRVCNPNSMWHIGKLVIISGIKQQVINCSDIQRWGINTSGEPTASGIYMHADVDFHAKNACMAESHRTEQHYFKQAPHPAQIHMFYYPYYNWMNQKNLWKTMFNTHESYAMDGVHKPLSAPVHRSHMMSCEDYEHWVVVAY'
#w = 'QCLMEMDHMWPQYEHLRIKWHPFMSKRHMVARTPSDFNKLFIHGRIRSNAGGQIFIGYRVKKAHQNDHIMPHQIDPQTSHHEIPDSKVDKDEHNHYAYGHKINWLEMERGWEWWTFTDIHGLPGCFSGDQICFKGAGFIFQSKRPMPVIMPCVRWDRSTLHRFVQCNCCKFDEVRFQWVMHEGPEAVADMVCGDIITAETGQSCNQKWDQKYSWCERENERDHCPNIPRGGWTLADVMFEVYGTTGEYNQTTMMGRHTIKKWQWDMFCKKATFRPLAFRAKSQEWMIQERCSQNVSWTREQHRIVQTWNPVKQSKMETEQTYECVCPGKQCVAESLTSLKFGIWENGSFPIRNEYFRTSDLDNWPFARIIEVEDEMLKLGACQLFGMLECHQGALTGPQDGRSQHPKIAMTFYCHVRSIHTDFIQYTRNDNHDCNMKGEYFCDNDGFWFAWEPMLPLSDDPLESKYMAHQHACIFTQHRACCYMLTYNPNSQMHENHKKLVRKANNGYAFCRTQDMTLHAPEGFRPGRWAHKYVLYHYYIHYRCDVLKFHACMSHLMLLSWDVGCMMWDAMPNGNCEFHIMATFNNFHMYKPIWRVCMINSMWHIGKLVIIRMDLGITDQVSDIQRWGINTSGEPTASGIFMHASVDFHAKVTEQHYFKQYRIYEYVPHPAQISSSMVHIVSFYYPYYNWMNQKNLCSGFVHAPSYAVHKPGWWILPHSAPVHHWHGTWIWMAQVAY'
#(a,b,c) = longest_common_subsequence(v,w, indel = 5, scoring = blosum62, verbose = True)
#print int(a)
#print b
#print c


pam250 = {'A': {'A': 2, 'C': -2, 'E': 0, 'D': 0, 'G': 1, 'F': -3, 'I': -1, 'H': -1, 'K': -1, 'M': -1, 'L': -2, 'N': 0, 'Q': 0, 'P': 1, 'S': 1, 'R': -2, 'T': 1, 'W': -6, 'V': 0, 'Y': -3}, 'C': {'A': -2, 'C': 12, 'E': -5, 'D': -5, 'G': -3, 'F': -4, 'I': -2, 'H': -3, 'K': -5, 'M': -5, 'L': -6, 'N': -4, 'Q': -5, 'P': -3, 'S': 0, 'R': -4, 'T': -2, 'W': -8, 'V': -2, 'Y': 0}, 'E': {'A': 0, 'C': -5, 'E': 4, 'D': 3, 'G': 0, 'F': -5, 'I': -2, 'H': 1, 'K': 0, 'M': -2, 'L': -3, 'N': 1, 'Q': 2, 'P': -1, 'S': 0, 'R': -1, 'T': 0, 'W': -7, 'V': -2, 'Y': -4}, 'D': {'A': 0, 'C': -5, 'E': 3, 'D': 4, 'G': 1, 'F': -6, 'I': -2, 'H': 1, 'K': 0, 'M': -3, 'L': -4, 'N': 2, 'Q': 2, 'P': -1, 'S': 0, 'R': -1, 'T': 0, 'W': -7, 'V': -2, 'Y': -4}, 'G': {'A': 1, 'C': -3, 'E': 0, 'D': 1, 'G': 5, 'F': -5, 'I': -3, 'H': -2, 'K': -2, 'M': -3, 'L': -4, 'N': 0, 'Q': -1, 'P': 0, 'S': 1, 'R': -3, 'T': 0, 'W': -7, 'V': -1, 'Y': -5}, 'F': {'A': -3, 'C': -4, 'E': -5, 'D': -6, 'G': -5, 'F': 9, 'I': 1, 'H': -2, 'K': -5, 'M': 0, 'L': 2, 'N': -3, 'Q': -5, 'P': -5, 'S': -3, 'R': -4, 'T': -3, 'W': 0, 'V': -1, 'Y': 7}, 'I': {'A': -1, 'C': -2, 'E': -2, 'D': -2, 'G': -3, 'F': 1, 'I': 5, 'H': -2, 'K': -2, 'M': 2, 'L': 2, 'N': -2, 'Q': -2, 'P': -2, 'S': -1, 'R': -2, 'T': 0, 'W': -5, 'V': 4, 'Y': -1}, 'H': {'A': -1, 'C': -3, 'E': 1, 'D': 1, 'G': -2, 'F': -2, 'I': -2, 'H': 6, 'K': 0, 'M': -2, 'L': -2, 'N': 2, 'Q': 3, 'P': 0, 'S': -1, 'R': 2, 'T': -1, 'W': -3, 'V': -2, 'Y': 0}, 'K': {'A': -1, 'C': -5, 'E': 0, 'D': 0, 'G': -2, 'F': -5, 'I': -2, 'H': 0, 'K': 5, 'M': 0, 'L': -3, 'N': 1, 'Q': 1, 'P': -1, 'S': 0, 'R': 3, 'T': 0, 'W': -3, 'V': -2, 'Y': -4}, 'M': {'A': -1, 'C': -5, 'E': -2, 'D': -3, 'G': -3, 'F': 0, 'I': 2, 'H': -2, 'K': 0, 'M': 6, 'L': 4, 'N': -2, 'Q': -1, 'P': -2, 'S': -2, 'R': 0, 'T': -1, 'W': -4, 'V': 2, 'Y': -2}, 'L': {'A': -2, 'C': -6, 'E': -3, 'D': -4, 'G': -4, 'F': 2, 'I': 2, 'H': -2, 'K': -3, 'M': 4, 'L': 6, 'N': -3, 'Q': -2, 'P': -3, 'S': -3, 'R': -3, 'T': -2, 'W': -2, 'V': 2, 'Y': -1}, 'N': {'A': 0, 'C': -4, 'E': 1, 'D': 2, 'G': 0, 'F': -3, 'I': -2, 'H': 2, 'K': 1, 'M': -2, 'L': -3, 'N': 2, 'Q': 1, 'P': 0, 'S': 1, 'R': 0, 'T': 0, 'W': -4, 'V': -2, 'Y': -2}, 'Q': {'A': 0, 'C': -5, 'E': 2, 'D': 2, 'G': -1, 'F': -5, 'I': -2, 'H': 3, 'K': 1, 'M': -1, 'L': -2, 'N': 1, 'Q': 4, 'P': 0, 'S': -1, 'R': 1, 'T': -1, 'W': -5, 'V': -2, 'Y': -4}, 'P': {'A': 1, 'C': -3, 'E': -1, 'D': -1, 'G': 0, 'F': -5, 'I': -2, 'H': 0, 'K': -1, 'M': -2, 'L': -3, 'N': 0, 'Q': 0, 'P': 6, 'S': 1, 'R': 0, 'T': 0, 'W': -6, 'V': -1, 'Y': -5}, 'S': {'A': 1, 'C': 0, 'E': 0, 'D': 0, 'G': 1, 'F': -3, 'I': -1, 'H': -1, 'K': 0, 'M': -2, 'L': -3, 'N': 1, 'Q': -1, 'P': 1, 'S': 2, 'R': 0, 'T': 1, 'W': -2, 'V': -1, 'Y': -3}, 'R': {'A': -2, 'C': -4, 'E': -1, 'D': -1, 'G': -3, 'F': -4, 'I': -2, 'H': 2, 'K': 3, 'M': 0, 'L': -3, 'N': 0, 'Q': 1, 'P': 0, 'S': 0, 'R': 6, 'T': -1, 'W': 2, 'V': -2, 'Y': -4}, 'T': {'A': 1, 'C': -2, 'E': 0, 'D': 0, 'G': 0, 'F': -3, 'I': 0, 'H': -1, 'K': 0, 'M': -1, 'L': -2, 'N': 0, 'Q': -1, 'P': 0, 'S': 1, 'R': -1, 'T': 3, 'W': -5, 'V': 0, 'Y': -3}, 'W': {'A': -6, 'C': -8, 'E': -7, 'D': -7, 'G': -7, 'F': 0, 'I': -5, 'H': -3, 'K': -3, 'M': -4, 'L': -2, 'N': -4, 'Q': -5, 'P': -6, 'S': -2, 'R': 2, 'T': -5, 'W': 17, 'V': -6, 'Y': 0}, 'V': {'A': 0, 'C': -2, 'E': -2, 'D': -2, 'G': -1, 'F': -1, 'I': 4, 'H': -2, 'K': -2, 'M': 2, 'L': 2, 'N': -2, 'Q': -2, 'P': -1, 'S': -1, 'R': -2, 'T': 0, 'W': -6, 'V': 4, 'Y': -2}, 'Y': {'A': -3, 'C': 0, 'E': -4, 'D': -4, 'G': -5, 'F': 7, 'I': -1, 'H': 0, 'K': -4, 'M': -2, 'L': -1, 'N': -2, 'Q': -4, 'P': -5, 'S': -3, 'R': -4, 'T': -3, 'W': 0, 'V': -2, 'Y': 10}}

#l = 'A  C  D  E  F  G  H  I  K  L  M  N  P  Q  R  S  T  V  W  Y'
#l = l.split(' ')
#l = [e for e in l if e != '' ]
#m = '2 -2  0  0 -3  1 -1 -1 -1 -2 -1  0  1  0 -2  1  1  0 -6 -3 -2 12 -5 -5 -4 -3 -3 -2 -5 -6 -5 -4 -3 -5 -4  0 -2 -2 -8  0 0 -5  4  3 -6  1  1 -2  0 -4 -3  2 -1  2 -1  0  0 -2 -7 -4 0 -5  3  4 -5  0  1 -2  0 -3 -2  1 -1  2 -1  0  0 -2 -7 -4 -3 -4 -6 -5  9 -5 -2  1 -5  2  0 -3 -5 -5 -4 -3 -3 -1  0  7 1 -3  1  0 -5  5 -2 -3 -2 -4 -3  0  0 -1 -3  1  0 -1 -7 -5 -1 -3  1  1 -2 -2  6 -2  0 -2 -2  2  0  3  2 -1 -1 -2 -3  0 -1 -2 -2 -2  1 -3 -2  5 -2  2  2 -2 -2 -2 -2 -1  0  4 -5 -1 -1 -5  0  0 -5 -2  0 -2  5 -3  0  1 -1  1  3  0  0 -2 -3 -4 -2 -6 -4 -3  2 -4 -2  2 -3  6  4 -3 -3 -2 -3 -3 -2  2 -2 -1 -1 -5 -3 -2  0 -3 -2  2  0  4  6 -2 -2 -1  0 -2 -1  2 -4 -2  0 -4  2  1 -3  0  2 -2  1 -3 -2  2  0  1  0  1  0 -2 -4 -2  1 -3 -1 -1 -5  0  0 -2 -1 -3 -2  0  6  0  0  1  0 -1 -6 -5  0 -5  2  2 -5 -1  3 -2  1 -2 -1  1  0  4  1 -1 -1 -2 -5 -4 -2 -4 -1 -1 -4 -3  2 -2  3 -3  0  0  0  1  6  0 -1 -2  2 -4  1  0  0  0 -3  1 -1 -1  0 -3 -2  1  1 -1  0  2  1 -1 -2 -3  1 -2  0  0 -3  0 -1  0  0 -2 -1  0  0 -1 -1  1  3  0 -5 -3  0 -2 -2 -2 -1 -1 -2  4 -2  2  2 -2 -1 -2 -2 -1  0  4 -6 -2 -6 -8 -7 -7  0 -7 -3 -5 -3 -2 -4 -4 -6 -5  2 -2 -5 -6 17  0 -3  0 -4 -4  7 -5  0 -1 -4 -1 -2 -2 -5 -4 -4 -3 -3 -2  0 10'
#m = m.split(' ')
#m = [int(e) for e in m if e != '' ]
#m = np.array(m).reshape(20,20)
#pam250 = {}
#for i in range(20):
#    d = {}
#    for j in range(20):
#        d[l[j]] = m[i, j]
#    pam250[l[i]] = d
        

assert longest_common_subsequence('PENALTY','MEANLY', indel = 5, scoring = pam250, verbose = True, local = True)[0] == 15

#v = 'AMTAFRYRQGNPRYVKHFAYEIRLSHIWLLTQMPWEFVMGIKMPEDVFQHWRVYSVCTAEPMRSDETYEQKPKPMAKWSGMTIMYQAGIIRQPPRGDRGVSDRNYSQCGKQNQAQLDNNPTWTKYEIEWRVQILPPGAGVFEGDNGQNQCLCPNWAWEQPCQWGALHSNEQYPNRIHLWAPMSKLHIKIEKSSYNRNAQFPNRCMYECEFPSYREQVDSCHYENVQIAFTIFSGAEQKRKFCSCHFWSNFIDQAVFSTGLIPWCYRRDDHSAFFMPNWNKQYKHPQLQFRVAGEGTQCRPFYTREMFTKVSAWRIAGRFAGPYERHHDAHLELWYQHHKVRTGQQLGIIWNNRDKTRNPCPFSAYYNKLPWWKINQNAFYNCLQNIAHSTHDETHEFNPVKCIDWLQGTMVPTECKKGFVHEKCECYRNPGPPLHDMYHQMEDIFGVRFDCLTGWKHLSDYNPCQERRNINDFYIFAYEIAPAVKNLVLSPQPLADATKKCAFNYTPLDQSPVVIACKWYIHQPICMLLIVLICAMDKYNAHMIVIRTTEGQQPMHACRMTEGPGMCMKEPLVTFTLPAQWQWPNHEFKYVYMYVLNYHLSQYTYTDEGHAGGQHYSFNVAVDVGMAWGHNRCYCQPACYSQQETQTRTIDYEKWQYMKHQAFKWGLWFCEQERHAWFKGQNRCEMFTAKMTRMGADSNLDQYKLMLAQNYEEQWEQPIMECGMSEIIEIDPPYRSELIFTFWPFCTYSPWQNLIKCRCNNVIEEMDQCVPLTFIGFGVKQAGGIQAWAFYKEEWTSTYYLMCQCMKSDKAQYPYEIILFWMQPMDTGEQEPPQQNMWIFLPHSWFFDWCCNAPWSEICSSRHDHGQCQDAFYPCELFTVFDDIFTAEPVVCSCFYDDPM'
#w = 'WQEKAVDGTVPSRHQYREKEDRQGNEIGKEFRRGPQVCEYSCNSHSCGWMPIFCIVCMSYVAFYCGLEYPMSRKTAKSQFIEWCDWFCFNHWTNWAPLSIVRTSVAFAVWGHCWYPCGGVCKTNRCKDDFCGRWRKALFAEGPRDWKCCKNDLQNWNPQYSQGTRNTKRMVATTNQTMIEWKQSHIFETWLFCHVIIEYNWSAFWMWMNRNEAFNSIIKSGYPKLLLTQYPLSQGSTPIVKPLIRRDQGKFWAWAQMWWFREPTNIPTADYCHSWWQSRADLQNDRDMGPEADASFYVEFWYWVRCAARTYGQQLGIIWNNRLKTRNPCPYSADGIQNKENYVFWWKNMCTKSHIAFYYCLQNVAHYTHDVTAEFNPVKCIDWLQGHMVLSSWFKYNTECKKLFVHEKCECYRMFCGVVEDIFGVRFHTGWKHLSTAKPVPHVCVYNPSVQERRNINDFYIFYEIAPAVKNLVLSAQPLHDYTKKCAFNYTPITITRIISTRNQIIWAHVVIACQFYSPHQMLLIELAMDKYCADMNVRRSTEGHQPMHACRSTFGPGMAAKEPLVTFTLVAFWQWPNHEFQYVYMYTEDKIIQIGPHLSNGCEMVEYCVDCYAKRPCYRAYSAEAQYWRMITEAEDYSYKTRNAIAATATVRGQYCHPFRWLGIVWMAHHDCFFANECGTICIPQMAEMRPPETTPYEIDIIFMMFWKEHMSTTILDVVGMYRPATFSHWHDAHHQCEPYLTPLMCQSKLVFDAAFTQVGVKGVWYHTEKLELMAGFNHMKFKKEEAQQSCFYWFQDCPDYDPPDAVRKTDEKHIRAHGEIWWLMRYYCMYHILHIASRHEWMHLRWDQACTNPGYELFEFIPWVLRRYVVYDKIRYNYSYRNSASMEFV'
#(a,b,c) = longest_common_subsequence(v,w, indel = 5, scoring = pam250, verbose = True, local = True)
#assert a == 1062
#print a
#print b
#print c



def edit_distance(v,w):
    l = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    edit_score = {}
    for i in range(20):
        d = {}
        for j in range(20):
            if (i==j): 
                d[l[j]] = 0
            else:
                d[l[j]] = -1
        edit_score[l[i]] = d
    (a,b,c) = longest_common_subsequence(v,w, indel = 1, scoring = edit_score, verbose = True, local = False)
    return -a
    
assert edit_distance('PLEASANTLY', 'MEANLY') == 5.0
assert edit_distance('TGCATAT', 'ATCCGAT') == 4.0

#v = 'WKYMHVLKKTMAWLYIEVVDKEEIRIDPLMPAWNHTFPRKWTQEQLIEPASQEWTHHPVPDIMEFIIILVTTYPGLPMPNAGDGVVSCNYHTQHVDKDTRKIPANNIISTFWLKNIQNNGLKEANTIRMPHGGPKGIHNCKNWKERSLAYVPLLYRFNQMYFHTQACCAWCLMWYIGPERCRGLVKTQLQNWMHIHKVHIWRWVGWKEHAQGKSWAKWNDGRGIAYRFIQFMSPDWEYYTNPPDSIIEYWRTEKMGCHAKTAHALEQGCLVEIIMEYNQSPHHWYTYPPGNTVDTYCMHGPYIQDIESFCFSQYRNNDQKCGNVTCWQDINCWTVGDRNPKDMTMTLAWEWPRTLWWTLGFQVNKELLMNSRMGTKIDRNAEINHCFFCGGRSRYVQNTMFVPWLHDSMQGHEDPMGLCTSRHAVLKIAMATDPMNHRHANYVWFCESHMNRRTAFFWVIHFWQWPKGNSNIMNIPNRGGHLMGSQHYIEIINCHATYRERKCHYNYYNGPPKGTRVRHSREKHWRIEGRMNNVYRHRFDSNAWFALWIWWLPECPHYYLCAHHPHCRNLHTKEENCDTWQYHFQQWRMMCDTEVFWWSHNQMVSGHEPWNLTCGTIQHSCLICFKTAMWTHGWHPDEQCMYRCGCHLPESEMVQGWMYACDASEFAVNTEHHVPNWVQNRMITQFWTNLHPQDFQIRTCGLPALSGGECTNHQTKCWAYWHKMTVCAIMEFFAYRPAPLISAGAVQMLIFRPMWHRRFLEHAPS'
#w = 'WKYMVTLKKTMAWLYIEVVDKLEITLCCDTTHRDPLMPHSPLFQWNHTKKVKSEMINPRRWTQEQLQNPARSGPFYIILVTTYPGLPMPNAGCVSCNYFDGSSWHITQHVDHDTANNIYSIQNNGLKEAIQTIRMPFGGPRMPVINRGIHLVPGQFMKCMYRFNQMYFGSPCVACCAWCLMWPERGLVWTQLQNWMHIHKPECHIWIWVGWKEHAQGKSWAKWNDGRGIAYRFIQFMIPDAWCVVCVHEYYRNPPNINGRFSIIEYWRTLDEPIEKMGCHAPHKCLLGMEIIMEYNPPGNTVDTICHHGPYIQDIESFGFSQIRNNDQKCYINCWTVGDRNPWGNITLCGARYLMWTMHGNHREGPQVNKELLFCQVHMHSNSQHWVRGYFQVELMVEGNHCNFLGGRSRYVQNTAFVPWLHDSMQGHHDPMGLCTSRHAVLKIAMAHDMNHRHMGCYVLYRVMMMTQTQWPKGNNIPNRGGHLMGSQHYIEINCHATYHYCQGERKCPSHTGPPKGTLVAHSGTSHWRIQDAQGRMNNVYRHRFDSNAHFALGNWQIWWLPKWKEYLCAHHPHCRNLHTKEDTWQYHFHQWRMMFPSCGLWDTEVFWWSHNQMVSGHEPWNVCGTIQHKCLICFCHYLDEFVMWTHGWHPDEQCMYRCFDLPEMVQGMPAFAVQGHSFLAYSTEHHVPNWVQFRQITQFWTNLHGQDKCASECGNPGLHQTKRWAYWHKFSYRPAPYIGAVQDLIFRWFRCGGLESAPS'
#print edit_distance(v,w)


def fitting_alignement(v, w):
    '''
    CODE CHALLENGE: 
    Solve the Fitting Alignment Problem.
    Input: Two nucleotide strings v and w, where v has length at most 1000 and w has length at most 100.
    Output: A highest-scoring fitting alignment between v and w. Use the simple scoring method in which
    matches count +1 and both the mismatch and indel penalties are 1.    
    '''
    n = len(v)
    m = len(w)
    indel = 1
    assert n > m
    s = np.zeros(shape = (n+1,m+1), dtype = np.float)
    for i in range(n+1):
        # free-ride taxi from source on v to take account of the fitting alignement
        s[i, 0] = 0
    for j in range(m+1):
        s[0, j] = -indel * j
    backtrack = np.chararray(shape = (n,m))
    for i in range(n):
        for j in range(m):
            score = (1 if v[i] == w[j] else -1)
            s[i+1, j+1] = max(s[i, j+1] - indel, s[i+1, j] - indel,
                s[i, j] + score)
            if s[i+1, j+1] == s[i, j+1] - indel:
                    backtrack[i, j] = '|'
            elif s[i+1, j+1] == s[i+1, j] - indel:
                    backtrack[i, j] = '-'
            elif s[i+1, j+1] == s[i, j] + score:
                    backtrack[i, j] = '/' if v[i] == w[j] else '*'
    lcs = []
    valign = []
    walign = []
    
    # for fitting alignement 
    # allow free-taxi ride to the sink on v
    # thus ensure s[i, j] is maximum over the last column of graph
    i = s[:,m].argmax()
    j = m

    smax = s[i, j]
    
    i -= 1
    j -= 1

#    print '------------'
#    print s
#    print backtrack
#    print local,i,j
#    print '------------'
    
    while (i >= 0 and j >= 0):
#        assert backtrack[i,j] != '*'
        if backtrack[i, j] == '|':
            walign.append('-')
            valign.append(v[i])
            i -= 1
        elif backtrack[i, j] == '-':
            walign.append(w[j])
            valign.append('-')
            j -= 1           
        else:
            lcs.append(w[j])
            walign.append(w[j])
            valign.append(v[i])
            j -= 1
            i -= 1
        
    lcs.reverse()
    valign.reverse()
    walign.reverse()
    lcs = ''.join(lcs)
    valign = ''.join(valign)
    walign = ''.join(walign)
    return (smax,valign,walign)

assert fitting_alignement('GTAGGCTTAAGGTTA', 'TAGATA')[0] == 2.0

#v = 'ATCGCGACAACGGTACCACATCAGAGCCGATCTTTTAGAGATACTTCAGCTGCCAATGGGTCAGCCGTAGCGAACGAACTCTAACCTGTGGCGGGTGCTCGACGACGGAGGTAGTGAACAGTCGTGACTTGATCCAAGATGAATCTGGTTATGTTTCGCTATATCACGATAGGGATTGGTGTATTTAACGGGGGTTATGTGACAACAACCGTGGTTGCCACCATAGCCGAGGCTCGAGCGTACCAAGGCCGGATATATACGGTTTTCTTTTTCGCAGCTAGTCTAGAACAGTATCGTTCAGACCCTGCTAAGGCGCTCGCAGCTCGGGCATCCCACCGGTATTATCATGAAACTGAATGACCTACTTGTTAGGTCCCTTGAACGACTTGCTATTTAGCTAGTTTGAGGTTGCCGTAGTCCTCAAGTAAAATTTAAGGGTATGCGTGACGATTAGTCGGAAGTTCAGCTCGATATAACCGGTAATCCTCTCCCAGATACGAGAACTGGCGGGGGGTTGCAAGTATATGCTAGACGACAATTGGGGTGGACAATTGGTGGACTAGAACTCTCGGAATCGAGTCGCTTTACTTTGCGAGATTTTGTTACCACCTCGTAACATGTTCGATCAAATACAGGAAGCCGGGTCTCGCAAGCATTAGTTCCCTTTTCGGCAGTCTTTGGGTCAATTTATGTGGTTTCAGTAGCTGCGAGACGAGGCGCGGAACGTTCTTCACACGACACCGGCATCTTATGCAGACATACCTGAGATGACTCGCGGTTATAGAGTAAATGACACCAGCGGCCTATAGAAGGTGCCGGCGTGCTTC'
#w = 'TCAAATATCGGGTCGCACCCAGTCCGGAAGCGCTCCGACGACCCTTAGGTGTATGACGTGGAGAAGCTTCGACGCGTTGAGTCAAGATAA'
#(a,b,c) = fitting_alignement(v, w)
#print int(a)
#print b
#print c


def overlap_alignement(v, w):
    '''
    CODE CHALLENGE: 
    Solve the Overlap Alignment Problem.
    Input: Two strings v and w, each of length at most 1000.
    Output: The score of an optimal overlap alignment of v and w, followed by an alignment of a suffix v' of
    v and a prefix w' of w achieving this maximum score. Use an alignment score in which matches count
    +1 and both the mismatch and indel penalties are 2.    
    '''
    n = len(v)
    m = len(w)
    indel = 2
    s = np.zeros(shape = (n+1,m+1), dtype = np.float)
    for i in range(n+1):
        # free-ride taxi from source on v to take account 
        # of the overlap-alignement-nature for a prefix-source
        s[i, 0] = 0
    for j in range(m+1):
        s[0, j] = -indel * j
    backtrack = np.chararray(shape = (n,m))
    for i in range(n):
        for j in range(m):
            score = (1 if v[i] == w[j] else -indel)
            s[i+1, j+1] = max(s[i, j+1] - indel, s[i+1, j] - indel,
                s[i, j] + score)
            if s[i+1, j+1] == s[i, j+1] - indel:
                    backtrack[i, j] = '|'
            elif s[i+1, j+1] == s[i+1, j] - indel:
                    backtrack[i, j] = '-'
            elif s[i+1, j+1] == s[i, j] + score:
                    backtrack[i, j] = '/' if v[i] == w[j] else '*'
    lcs = []
    valign = []
    walign = []
    
    # for overlap alignement 
    # allow free-taxi ride from the end of source to the sink on v
    # thus ensure s[i, j] is maximum over the last row of graph
    i = n
    j = s[n,:].argmax()

    smax = s[i, j]
    
    i -= 1
    j -= 1

#    print '------------'
#    print s
#    print backtrack
#    print local,i,j
#    print '------------'
    
    while (i >= 0 and j >= 0):
#        assert backtrack[i,j] != '*'
        if backtrack[i, j] == '|':
            walign.append('-')
            valign.append(v[i])
            i -= 1
        elif backtrack[i, j] == '-':
            walign.append(w[j])
            valign.append('-')
            j -= 1
        else:
            lcs.append(w[j])
            walign.append(w[j])
            valign.append(v[i])
            j -= 1
            i -= 1
        
    lcs.reverse()
    valign.reverse()
    walign.reverse()
    lcs = ''.join(lcs)
    valign = ''.join(valign)
    walign = ''.join(walign)
    return (smax,valign,walign)
    
assert overlap_alignement('PAWHEAE', 'HEAGAWGHEE')[0] == 1.0


#v = 'GCTATAAGAATAAACCACTAGATCACCTCCGGCTCGCTCACTCCTGATCATGGTTCGTGCTAACATCGCGCCGCGCTGACGCCGAATCGTTCGTAGGAGACAAGTCGACGACCTCATCTACAGGCAAAAGTTAAATTAGCTCTCGGCTAGATGTGACAATCGGAACCCTGCACCCTGCGTAATAGGGTAAATAGTCGGGAGTTGATGCACACACCTAGATATTGGCTGAATGACAGACTGCCATTCCTGCACTGGAAAGTAGAGTGCATATGTTTCGTGAGATTATGCAGGCTCTACGGTTATACTGGGCTCCACGGATTCGACCGGTACTGTTGATTGAAGACTCTTCTATAGAGGCTCTAACCGCGGAGGCCGCAACCAATCGACAATGAAGCACCCGTCGTCGGTATCGTTGGGAAGGACGACACCGTAAGGGCAGACTTTATCGTGACCCGTCTGCTTGCTAGAAAAGCCCTGGCGTTTGTACAACGTCCGTGCAGAATTAGCGTTTTTCTCAGGAAAGATGAGGGGGTTGATCATCATCTCGTTTCGCACGGGTCAAGCGCATTTTCCTACTGTTTTGGACACAGTACGTCTTCCACTGATCTCATACGGACATTACCAGCACCCTTTTGTACCTGTCGTAACTTGTGCCATTCTAGGCCCGTTTTCACTTGCGCTTATGATCATGGTTCCGCTGATCTATATGGGCCGGGTAGGGCACTCCCAGATGAAGGGGAGTAATGGTAGCCGGATCCAAGTGACGCGCCCTAGCGGCTCCGGAGTTTGATAGACGTCGTGCTATGGAGCGTTGGAGCGACAACGCGCTCGTGCTCTGGAAGGTCGCTGCTGATCCGTAA'
#w = 'TACTGGTCCTGACCCACCTCACTTTGATGTCCCCTTTTCTCGTTTGCGCATCAAGATCTGGCCCGCAACTATTGGCCGTGAAAGGCACTCATCAATAAAGACAGTACTCACGCGGTCGGATCCAAATGCGCGCACCGAGCGGCCCAGGAGTTGATAGCGTCGAGTAACCTATTAGGACTCGAGGCAACTCGCGCTCTCTCAGGAGGCTCGCCTGCTAGTCCGTGAACGACGGATCTTTGGTGCTGCCTTCCTATCATGACATTGCCTAATAACGAGCGGCACCTACTCCCAGGTCTTTGAAGGGATGGCTTGTTTACCCCGATTCCGAGAAATAGAGATGACTCCTAAGGAAGTAATGAAGGAAGTTCAGTGGTATGGGTATCGTTTAGTTTGCCAGGGAGATTGCCCATAACCTAAGTCCCTAATACAGCAGTAGATCTCACCATAGATGTAGGAAAGCACAGTGATTTAGACGCTTAGCCAAATACAAAGGAATGTACCCCCTCCTAACACTGAGCACCGCTTATTTACTAGTATACTCAGAGTGTGGAGCGCTGAACGTTGTGTCAACAAGAACATAAGCCGCCGTGAATGAATTTGTGAAGGGGAGTGATCATGGTTTTACTCGTGGTAGATTTGGGCAGAACCTGATTCCTCACGTGTGAATGTAATTGAAGCTGACTCCCACACATACAGGCACGATTCTTTTAGATGATGTTTTAGGAAGCGCATTTCGTATTAACACTGCCTTGCATTTGATAACCATCACTTGTTCATTACATGATCCCATAGGGCCGTGTTGTTACTTTCGTGTTAGTCGAGCAGTATGACCACCTTTTCGGCGCTTGATATGCCTCAAGACGTGCGATTCAAGGAATCAAACAAATGAACGCCGCACTGGATGACTGGG'
#(a,b,c) = overlap_alignement(v, w)
#print int(a)
#print b
#print c

#v = 'ATTATGCCGTACAAACCCGGTAACTCTGTACTTTATCTACTCAGTCGGTCTATCGTAGGGACGTCTGAAAAATCTCATGGAGTGCCCTCACTAAAGTTTTCCGCATCTTTCATAGGCGCCGCCAGCGTCCGCTACCCCTCCTCTTCATCGGAGAACATTATCGGAGCAGTGCTAGACCGGGCGTCGACCTGGCGGTGTGCCAGTATTCCTATGTCTGTCTAAACGTAGCTCTATGGGATCGGCGTGTTGGCCCGAGAAAGTCTGACTGCCGAAACGCCTCAGAGACAGCCCCTGCCATATTGAACGTTCGCTCAGAGACGGCGTTATGCCGACTAGAGAGCCCAAGGGCGGGTCCCGACGTCGATGCGCATGTGTACTAACGACATTTTATGCTAGAACAGGCACGACGGTCGCCCAGACAGCCCGGGGAAGTTAGAATGATGGGGTACCAGATTTCGGGAATATGTTTCCAGACACTGTTCCCTGTCATTGGGTTCTCGTTCCAGACGCGTTGAGACTGGCAAGACCTACTTAAAAGTTCTCGCGGCCAGTTATTGAAGTAACCACTCCAGCAAGACCTGCCTACTTCTGTGGAGTGTACCTTCCCGAGAGGTCTAAACGGGTAGTGCCAGTTACTAATTTGCGGTTTAAGGCCTATAGCTCACAATGCTTGCACACTTTAGCATTTATTTTTGAGTATTGGTGCGGCGGGAGCGACCGTGCACTAGGAAGACACCGCCCACCTTATCCAGGTCAAGTCGGGCCTTAGACAGTATAGATTAGAAATCATTGAGTAGTTTTATAGTAGCTTCAGTCCTCGCGCGATACTGACGAGCATCGGATAGAAGACCCAAAGTTGTTATTCCGGGAAAATGAAGGAAATTTCTCTATGTGTCTACACATAG'
#w = 'TTCATTTTTGAATTTGCTGCTGGAGAAGGACAACTGTGCACAGGAAGACAGCACTAGCCTATCAGGTCAAGCGGTGCCCTTCGGACAGTAAGAAAAAAATCTTTTGAGTGTGCCAGTTGCAGTGCCTGGATTCCATGATCCTGACGAGCAGGCGAATGAAAGCCAAGTTGTTATACCGGGAATATGGAGGACAAGTATTCCGCTAGTATCCTAGCAGCCAGCTGCAGCATCTTTACTTCCTAGATAAGACGCTCCTTCGTCAAAACGCAGCCATTTTGCCCTCTAGATCAACGCTCTAAAGACGGCTATACTCTATATACAACTGACCTCCCCTGTCCTGTTGCTTCCGACGAAGCTGACTCTCCTGTTGGGTCCCTGTGTTTTATGGGGTTAGTAAACTTACCGCCGAGTTCCTCTCTAATTTAACTGGACGGTACGTATATATCTGCTTCGTCTACTCGCGTCGAGCCGGAGCGCCTCGAGGTGGCGGCTGTGCACATGTAGTATAGGGTTCTGCGCCCGAGCCACGACCGCGTTTTATCCAACATTGACAACGGGGGAACCCCGACATCTCATAGACAAAATAAATATCAACAATTGGCGTGTCGACTATGCTATACAATTTAGGTGGATTGGGCTGGTGTACCATGCCACACTATTATGTAAACGTTCATCTATAATTCCAGGATGCCCCCTGAGCGGTCCGGACCGCATAAGGACATTCTGCAAGGTGATGTAAACTGAACAAAAGTCCCGTACTTTGCCCGCTTCCCCTTAGTGGTTAGAGCTAGACTTTGCTCTATTTCAAATAGGGCCATAACTCCCGAAGGCCTGCAGGTGGTTATCCGCGCCGAT'
#(a,b,c) = overlap_alignement(v, w)
#print int(a)
#print b
#print c

def with_gap_alignement(v, w, scoring = blosum62):
    '''
    CODE CHALLENGE: 
    Solve the Alignment with Affine Gap Penalties Problem.
    Input: Two amino acid strings v and w (each of length at most 100).
    Output: The maximum alignment score between v and w, followed by an alignment of v and w
    achieving this maximum score. Use the BLOSUM62 scoring matrix, a gap opening penalty of 11, and
    a gap extension penalty of 1.
    '''
    n = len(v)
    m = len(w)
    sigma = 11
    epsilon = 1
    middle = np.zeros(shape = (n+1,m+1), dtype = np.float)
    upper = np.zeros(shape = (n+1,m+1), dtype = np.float)
    lower = np.zeros(shape = (n+1,m+1), dtype = np.float)
    for i in range(n+1):
        middle[i, 0] = -sigma*i
        lower[i, 0] = -epsilon*i
    for j in range(m+1):
        middle[0, j] = -sigma*j
        lower[0, j] = -epsilon*j
    # backtrack matrix has 3 level : level 0 is lower '|', level 1 is middle '\', level 2 is upper '-'
    # backtrack in level 0 is a vertical gap e.g insertion of '-' for W-alignement
    # backtrack in level 2 is an horizontal gap e.g insertion of '-' for V-alignement
    backtrack = np.chararray(shape = (n,m,3))
    for i in range(n):
        for j in range(m):
            score = scoring[ v[i] ][ w[j] ]
            lower[i+1, j+1] = max(lower[i, j+1] - epsilon, middle[i, j+1] - sigma)
            upper[i+1, j+1] = max(upper[i+1, j] - epsilon, middle[i+1, j] - sigma)
            middle[i+1, j+1] = max(lower[i+1, j+1], upper[i+1, j+1],  middle[i, j] + score)
            
            if lower[i+1, j+1] == lower[i, j+1] - epsilon:
                # continuing vertical gap
                backtrack[i, j, 0] = '|'
            else:
                #opening vertical gap
                backtrack[i, j, 0] = '+'
    
            if upper[i+1, j+1] == lower[i+1, j] - epsilon:
                # continuing horizontal gap
                backtrack[i, j, 2] = '-'
            else:
                #opening vertical gap
                backtrack[i, j, 2] = '+'
                            
            if middle[i+1, j+1] == lower[i+1, j+1]:
                # vertical gap closing
                backtrack[i, j, 1] = '|'
            elif middle[i+1, j+1] == upper[i+1, j+1]:
                # horizontal gap closing
                backtrack[i, j, 1] = '-'
            elif middle[i+1, j+1] == middle[i, j] + score:
                backtrack[i, j, 1] = '/' if v[i] == w[j] else '*'

    valign = []
    walign = []
    
    i = n
    j = m

    smax = middle[i, j]
    
    i -= 1
    j -= 1
    level = 1

#    print '------------'
#    print lower
#    print middle
#    print upper
#    print backtrack[:,:,0]
#    print backtrack[:,:,1]
#    print backtrack[:,:,2]
#    print '------------'
    
    while (i >= 0 and j >= 0):
        if (level == 1):
            if backtrack[i, j, level] == '|':
                level = 0
            elif backtrack[i, j, level] == '-':
                level = 2
            else:
                walign.append(w[j])
                valign.append(v[i])
                i -= 1
                j -= 1
        elif (level == 0):
            # back tracking a vertical gap
            walign.append('-')
            valign.append(v[i])
            if (backtrack[i, j, level] == '+'):
                # back tracking a gap opening
                level = 1
            i -=1
        elif (level == 2):
            # back tracking an horizontal gap
            walign.append(w[j])
            valign.append('-')
            if (backtrack[i, j, level] == '+'):
                # back tracking a gap opening
                level = 1
            j -=1
        
    valign.reverse()
    walign.reverse()
    valign = ''.join(valign)
    walign = ''.join(walign)
    return (smax,valign,walign)
    
assert with_gap_alignement('PRTEINS', 'PRTWPSEIN')[0] == 8.0

v = 'QCSYYMHQTSGSGPSRKDDIIMCRHKFIPCEIRTMTYKCWCMTTIYRLPMHSAALFFRFGHIWFMGGAYRTRAQLYTLNFSITK'
w = 'QCSYYMNFFLQYTSGSGPKRKDDIINCRHQFIPCEIRTMTYKCWCMTTIYRLPMHSAAHMFRFFHIWFMGGAYRTRAQLDTLIFSDTK'
(a,b,c) = with_gap_alignement(v, w)
print int(a)
print b
print c


#v = 'YHFDVPDCWAHRYWVENPQAIAQMEQICFNWFPSMMMKQPHVFKVDHHMSCRWLPIRGKKCSSCCTRMRVRTVWE'
#w = 'YHEDVAHEDAIAQMVNTFGFVWQICLNQFPSMMMKIYWIAVLSAHVADRKTWSKHMSCRWLPIISATCARMRVRTVWE'
#(a,b,c) = with_gap_alignement(v, w)
#print a
#print b
#print c