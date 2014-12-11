# -*- coding: utf-8 -*-
"""
Created on Mon Dec 08 22:45:00 2014

@author: ngaude
"""

from itertools import product


def composition(k, text):
    '''
    Solve the String Composition Problem.
    Input: An integer k and a string Text.
    Output: Compositionk(Text), where the k-mers 
    are written in lexicographic order.
    '''
    kmers = []
    for i in range(len(text) - k + 1):
        kmers.append(text[i:i+k])
    return sorted(kmers)

def genome_path(path):
    '''
    String Spelled by a Genome Path Problem. 
    Reconstruct a string from its genome path.
    Input: A sequence of k-mers Pattern1, … ,
    Patternn such that the last k - 1 symbols of Patterni are
    equal to the first k-1 symbols of Patterni+1 for 1 ≤ i ≤ n-1.
    Output: A string Text of length k+n-1 such that 
    the i-th k-mer in Text is equal to Patterni  (for 1 ≤ i ≤ n).
    '''
    return ''.join([e[0] for e in path])+path[-1][1:]

def overlap(patterns):
    '''
    Solve the Overlap Graph Problem (restated below).
    Input: A collection Patterns of k-mers.
    Output: The overlap graph Overlap(Patterns), 
    in the form of an adjacency list.
    '''
    # build a prefixing pattern dict
    dprefix = {}
    ladj = []
    for e in patterns:
        prefix = e[:-1]
        dprefix.setdefault(prefix, []).append(e)
    for e in sorted(patterns):
        suffix = e[1:]
        for ee in dprefix.get(suffix, []):
            ladj.append((e,ee))
    return ladj

def universal_string_brute_force(k):
    def bin2string(d):
        if d == 0:
            return ''
        else:
            return bin2string(d>>1) + ('1' if d&1 else '0')
    ref = []
    for i in range(pow(2,k)):
        s = bin2string(i)
        s = '0' * (k - len(s)) + s
        ref.append(s)
    ref.sort()
    ustring = []
    ustringlen = pow(2,k) + (k-1)
    for i in range(pow(2,ustringlen)):
        s = bin2string(i)
        s = '0' * (ustringlen - len(s)) + s
        if (composition(k,s) == ref):
            ustring.append(s)
    return ustring

def universal_circular_string(k):
    '''
    Solve the k-Universal Circular String Problem.
    Input: An integer k.
    Output: A k-universal circular string.
    '''
    kmers = [''.join(x) for x in product('01', repeat=k) ]
    g = debruijn_from_kmer(kmers)
    c = eulerian_cycle(g)
    s = genome_path(c)[k-1:]
    # push back zero to garantee the zero node 
    # for not being cut when wrapping
    while (s[0]=='0'):
        s = s[1:]+'0'
    # btw,
    # it seems that the checkper need the circular string 
    # to start with the zero node e,g '0'*k
#    print 'g',g
#    print 'c',c
#    print 's',s
    i = s.index('0'*k)
    return s[i:]+s[:i]
    
def debruijn(k, text):
    '''
    Construct the de Bruijn graph of a string.
    Input: An integer k and a string Text.
    Output: DeBruijnk(Text).
    '''
    # build pattern list of len(text)_k+1 kmers from text
    patterns = composition(k, text)
    #
    return debruijn_from_kmer(patterns)

def debruijn_from_kmer(kmers):
    '''
    Construct the de Bruijn graph from a set of k-mers.
    Input: A collection of k-mers Patterns.
    Output: The adjacency list of the de Bruijn graph DeBruijn(Patterns).
    '''
    g = []
    # build a prefixing pattern dict
    dprefix = {}
    for e in kmers:
        prefix = e[:-1]
        dprefix.setdefault(prefix, []).append(e[1:])
    # build lexicographically sorted adjacency list
    for k in sorted(dprefix.keys()):
        g.append( (k,sorted(dprefix[k])) )
    return g

def eulerian_cycle(adjacency_list):
    #build adjacency dict from adjacency list
    dadj = {}
    for (k, v) in adjacency_list:
        dadj[k] = dadj.get(k,[]) + v[:]
    #get next available node from graph given current node
    def next_node(curr):
        nlist = dadj.get(curr,None)
        if (nlist is None):
            return None
        else:
            nnode = nlist[0]
            nlist.remove(nnode)
            if (nlist == []):
                dadj.pop(curr)
            return nnode
    # get a graph inner cycle
    def inner_cycle(cycle):
        return iterative_inner_cycle(cycle) 
        
    def recursive_inner_cycle(cycle):
        '''
        nice recursive implementation, 
        but python doesn't appreciate that much when depth > 1000...
        RuntimeError: maximum recursion depth exceeded 
        while calling a Python object
        '''
        nn = next_node(cycle[-1])
        if nn is None:
            return cycle
        cycle.append(nn)
        if nn == cycle[0]:
            return cycle
        return recursive_inner_cycle(cycle)

    def iterative_inner_cycle(cycle):
        '''
        classic iterative,
        and less functionnal-programming approach
        '''
        while True:
            nn = next_node(cycle[-1])
            if nn is None:
                return cycle
            cycle.append(nn)
            if nn == cycle[0]:
                return cycle

    def expand_cycle(cycle):
        # search for an expandable node
        for n,v in enumerate(cycle):
            if v in dadj:
                prefix = cycle[:n]
                suffix = cycle[n+1:]
                icycle = inner_cycle([v])
#                print 'cycle',cycle,'expand',v,'id',n
#                print 'prefix',prefix
#                print 'icycle',icycle
#                print 'suffix',suffix
#                print '--------------'
                return prefix+icycle+suffix  
    # assert adjacency dict is not empty
    assert dadj
    # start with a single-node cycle
    euler = [dadj.keys()[0]]
    prevlen = currlen = 0
    while dadj and euler:
        currlen = len(euler)
        if (currlen <= prevlen):
            # if expansion failed whilst edges must still be visited 
            # then no eulerian path can be found
            return None
        prevlen = currlen
        euler = expand_cycle(euler)
    return euler

def rotate_cycle(cycle,value):
    '''
    rotate input cycle 
    such as returned cycle starts with 
    the first encountered node of given value 
    '''
    assert value in cycle
    i = cycle.index(value)
    if (i==0):
        # no rotation indeed
        return cycle
    else:
        return cycle[i:]+cycle[1:i]+[value]   

def nearly_balanced(adjacency_list):
    '''
    return edge that will balance perfectly 
    given graph assumed to be nearly balanced
    Input : nearly balanced graph
    Output : balancing_edge
    '''
    ind = {}
    outd = {}
    for (k, v) in adjacency_list:
        outd[k] = len(v)
        for kk in v:
            ind[kk] = ind.get(kk,0)+1
    end = [(k, v-outd.get(k,0)) for k,v in ind.iteritems() if v > outd.get(k,0) ]
    beg = [(k, v-ind.get(k,0)) for k,v in outd.iteritems() if v > ind.get(k,0) ]
    if (len(end) ==  1) and (end[0][1] == 1) and \
        (len(beg) ==  1) and (beg[0][1] == 1):
        return (end[0][0], beg[0][0])
    return None

def eulerian_path(adjacency_list):
    '''
    Solve the Eulerian Path Problem.
    Input: The adjacency list of a directed graph that has an Eulerian path.
    Output: An Eulerian path in this graph
    '''
    # check the graph is nearly balanced
    edge = nearly_balanced(adjacency_list)
    if (edge is None):
        # unbalanced : no eulerian path can be found
        return None
    # add the extra balancing edge and extract an eulerian cycle
    cycle = eulerian_cycle(adjacency_list+[(edge[0], [edge[1]])])
    if (cycle is None):
        # unconnected : no eulerian path can be found
        return None
    #locate precisely the edge transition with the cycle
    for i,v in enumerate(cycle[:-1]):
        if (v == edge[0]) and (cycle[i+1] == edge[1]):
            return cycle[i+1:]+cycle[1:i+1]
        

def genome_reconstruction(kmers):
    '''
    Solve the String Reconstruction Problem.
    Input: An integer k followed by a list of k-mers Patterns.
    Output: A string Text with k-mer composition equal to Patterns. 
    (If multiple answers exist, you may return any one.)
    '''
    return genome_path(eulerian_path(debruijn_from_kmer(kmers)))


fname = 'C:/Users/ngaude/Downloads/dataset_203_6.txt'
kmers = list(l[:-1] for l in open(fname))[1:]
genome = genome_reconstruction(kmers)
with open(fname+'.out', "w") as f:
        f.write(genome)

assert genome_reconstruction(['CTTA','ACCA','TACC','GGCT','GCTT','TTAC'])=='GGCTTACCA'
assert eulerian_path([(0,[2]),(1,[3]),(2,[1]),(3,[0,4]),(6,[3,7]),(7,[8]),(8,[9]),(9,[6])]) == [6, 7, 8, 9, 6, 3, 0, 2, 1, 3, 4]
assert rotate_cycle(eulerian_cycle([(0,[3]), (1,[0]), (2,[1,6]), (3,[2]), (4,[2]), (5,[4]), (6,[5,8]), (7,[9]), (8,[7]), (9,[6])]),0) == [0, 3, 2, 6, 8, 7, 9, 6, 5, 4, 2, 1, 0]
assert debruijn(4, 'AAGATTCTCTAAGA') == [('AAG', ['AGA', 'AGA']), ('AGA', ['GAT']), ('ATT', ['TTC']), ('CTA', ['TAA']), ('CTC', ['TCT']), ('GAT', ['ATT']), ('TAA', ['AAG']), ('TCT', ['CTA', 'CTC']), ('TTC', ['TCT'])]
assert universal_string_brute_force(2) == ['00110', '01100', '10011', '11001']
assert overlap(['ATGCG','GCATG','CATGC','AGGCA','GGCAT']) == [('AGGCA', 'GGCAT'), ('CATGC', 'ATGCG'), ('GCATG', 'CATGC'), ('GGCAT', 'GCATG')]
assert genome_path(['ACCGA','CCGAA','CGAAG','GAAGC','AAGCT']) == 'ACCGAAGCT'
assert composition(5, 'CAATCCAAC') == ['AATCC', 'ATCCA', 'CAATC', 'CCAAC', 'TCCAA']


#fname = 'C:/Users/ngaude/Downloads/dataset_203_5.txt'
#ladj = [ ( int(l.split(' -> ')[0]), map(int,l.split(' -> ')[1].split(',')) ) for l in open(fname) ]
#ep = eulerian_path(ladj)
#with open(fname+'.out', "w") as f:
#    f.write('->'.join(map(str,ep)))

#   
#fname = 'C:/Users/ngaude/Downloads/dataset_203_2.txt'
#ladj = [ ( int(l.split(' -> ')[0]), map(int,l.split(' -> ')[1].split(',')) ) for l in open(fname) ]
#ec = eulerian_cycle(ladj)
#with open(fname+'.out', "w") as f:
#    f.write('->'.join(map(str,ec)))
#

#fname = 'C:/Users/ngaude/Downloads/dataset_200_7.txt'
#kmers = list(l[:-1] for l in open(fname))
#g = debruijn_from_kmer(kmers)
#with open(fname+'.out', "w") as f:
#    for (e,ee) in g:
#        l = e + ' -> ' + ','.join(ee) + '\n'
#        f.write(l)


#fname = 'C:/Users/ngaude/Downloads/dataset_199_6.txt'
#(k, text)  = (l[:-1] for l in open(fname))
#g = debruijn(int(k), text)
#with open(fname+'.out', "w") as f:
#    for (e,ee) in g:
#        l = e + ' -> ' + ','.join(ee) + '\n'
#        f.write(l)

#fname = 'C:/Users/ngaude/Downloads/dataset_198_9.txt'
#patterns = list(l[:-1] for l in open(fname))
#ladj = overlap(patterns)
#with open(fname+'.out', "w") as f:
#    for (e,ee) in ladj:
#        adji = e + ' -> ' + ee + '\n'
#        f.write(adji)

#p = []
#for l in open('C:/Users/ngaude/Downloads/dataset_198_3.txt'):
#    p.append(l[:-1])
#o = genome_path(p)
#print o

#fname = 'C:/Users/ngaude/Downloads/dataset_197_3.txt'
#(k, text)  = (l[:-1] for l in open(fname))
#o = '\n'.join(composition(int(k),text))
#with open(fname+'.out', "w") as f: f.write(o)