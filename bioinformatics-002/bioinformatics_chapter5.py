# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 20:08:03 2014

@author: ngaude
"""

import numpy as np

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

def longest_common_subsequence(v,w):
    '''
    CODE CHALLENGE: solve the Longest Common Subsequence Problem.
    Input: Two strings v and w
    Output: A longest common subsequence of v and w. 
    (Note: more than one solution may exist, in which case you may output any one.)
    '''
    n = len(v)
    m = len(w)
    s = np.zeros(shape = (n+1,m+1), dtype = np.float)
    backtrack = np.chararray(shape = (n,m))
    for i in range(n):
        for j in range(m):
            s[i+1, j+1] = max(s[i, j+1], s[i+1, j], s[i, j] + (1 if v[i] == w[j] else 0))                
            if s[i+1, j+1] == s[i, j+1]:
                    backtrack[i, j] = '|'
            elif s[i+1, j+1] == s[i+1, j]:
                    backtrack[i, j] = '-'
            elif s[i+1, j+1] == s[i, j] + (1 if v[i] == w[j] else 0):
                    backtrack[i, j] = '/' if v[i] == w[j] else '*'
    lcs = []
    i = n-1
    j = m - 1
    while (i >= 0 and j >= 0):
        assert backtrack[i,j] != '*'
        if backtrack[i, j] == '|':
            i -= 1 
        elif backtrack[i, j] == '-':
            j -= 1
        else:
            lcs.append(w[j])
            j -= 1
            i -= 1
    lcs.reverse()
    
#    print 's',s.shape
#    print s
#    print 'bt',backtrack.shape
#    print backtrack

    return ''.join(lcs)
   
def dag_longest_path(source, sink, edges):
    '''
    Input: 
    the source node of a graph, 
    the sink node of the graph,
    followed by a list of edges in the graph.    
    Output: longest path from source to sink as a the list of nodes.
    '''

assert len(longest_common_subsequence('AACCTTGG','ACACTGTGA'))== 6

print longest_common_subsequence('AACCTTGG','ACACTGTGA')

v = 'ATAGTCTGCGTCCACCAAGAACTCGCGATACGGCTTGGGTGAGAGAAGAAATTGTTGGACTGTCGTGGGTATAACATCCGGGTCGAGGCGAAGGTCCGCTAGTTATCCCCTGGCGTGGCTAAATAGCAATTTTGCAGAATGCCTTATCTATCTTCAGCCATGCGCTTAACTAAATATTTTCAGTATAGTGTTACTAACGAGGAAGTCAAGTGCGTCTAATTTATTGAATAGAACTACTAGTTTTCCCCATAACGCCTAACTTTTAGAATACGATCTCTTTAGGATTAGAGATTTGCCAGGGCACACCGGGGTTGAGAGGGAAGGGGTTTGTACCTCGTGCGGCCTCAGGGGGGCGGTGTTGTCAGAATCTATCCGCACCCGCAGACCTATATTCCTACGTGGCAAGCCTGCATTTGTTAGCTGGTCCGCCTCGCGCAGGCGTCCGTCGGACCAGTGTCGAGAGCCCAATGCATTAGCAGTTCTTTGCCGTTATGGTTCTAAATGAGAAAGCACGTAGCCTTCGAATCAGTTTCCCGATTACTGGCTCCTCGACATAAAAAATCTGTCTCGAGTAGCATTCTGACGATAGGTTAGGGTGCCTGGAGTGAATTTAGCTGCATAATACCTTTGCGACCGTAGAAGACCCGGCCTGGATAGTGATCACCGCTGCTCCCCCATCTGGTTTCCTAGCTAATTGGGTCAGTGACTCTCCTCTCACTTGGGTTCTTCTTCTACGGAATCTGCGAAACACTGCAATATGGACCACAGAGCTTAAAGTATAGAGTCGCGCCAGTCACCACTCTGGTGATCAGTGAAGGGAATGATCACTCCAAACTGAACTCGCCGGTTTTAGTCTTGGCGGCCTTGTACGCTCACGGGTGAAGTAAGTACTAG'
w = 'TTTCGCCGGGGCCTACGACAACTACACCACTTTTATTTCTCCTTATTGAATCTATGTTATCTGCGATAACGAGCAAGGGCACGGTCCGGTAGAATAAACTGGATGCATTAGGCACCATATTGAATTCAGGTGGGTACCGGTGTCCCCTCGTCCGGCGGACTAGCGGATGGCGGACATCAAGGGGGCGCACTGCCCAAGGGAAACACTCCGGGCATGCACCTCGATCACAGGTGTCTCACTCGTCACAGGTCTGCAAGTACCACGACAATTTCCTTCTCCTGGTGGATTCTTAGGGAGGAGTCCACTATACATACTCGTATCAGACTACTCAGGCCACCTCCTTAGGCGAAGGATATTCACCGGTGTGCGTTTGCAGACCCGACTGACCGTCTATGCCAGCCCTCAGTTATGCAAAATTTTACGATTTCACTGTCTACGTGGCCGGACTGGCCTGAACGGCAAAAAGGCAAGAGGATGAGCGCATATATGGGGTCCAGTTACATCACCAAATTCTAGGGCATTACTAAGCTATGACTGGCTACAGGTGTTGCGGTACCTCCCAGTAGATATTTGAGCGGAAACCGTATGACTGCAAGTTCGGATTAAGAGCCGCCGACGCTATCAGGCGCAGCGACAAATCGGAACGCGTCAGACGAACCAGGAAGAACGGCTGCTAGCCTTTACAACCTGATCAGCCCGTTGAGCGAAATGAATTACGGCTCGGGGGCAACTCTGACAGACCTCAGGTTTAAGCCTAAAACGTAGGACCCTTGAGCTTTCGCGCTATCTCCTAGAAGAAGGGAACCAGTGTCGCCTGTACGTTGCTGTAACCGTTTTTTGTACACCACAAGTCAGCCTCGCCCTCATTTTACTATGTCGGTGGCT'


print longest_common_subsequence(v,w)

assert manhattan_tourist(4,4,((1,0,2,4,3),(4,6,5,2,1),(4,4,5,2,1),(5,6,8,5,3)),((3,2,4,0),(3,2,4,2),(0,7,3,3),(3,3,0,2),(1,3,2,2))) == 34

#fname = 'C:/Users/ngaude/Downloads/dataset_261_9.txt'
#lines = list(l.strip() for l in open(fname))
#n = int(lines[0].split(' ')[0])
#m = int(lines[0].split(' ')[1])
#down = map(lambda l : map(int, l.split(' ')), lines[1:1+n])
#assert lines[1+n] == '-'
#right = map(lambda l : map(int, l.split(' ')), lines[2+n:2+n+n+1])
#print manhattan_tourist(n, m, down, right)