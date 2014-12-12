# -*- coding: utf-8 -*-
"""
Created on Mon Dec 08 22:45:00 2014

@author: ngaude
"""

from itertools import product
import copy

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
    
def pair_composition(k, d, text):
    kmers = composition(2*k+d,text)
    pairs = map(lambda  e: (e[:k],e[-k:]), kmers)
    return pairs

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



def all_eularian_cycles(adjacency_list):
    #build adjacency dict from adjacency list
    original_graph = {}
    for (k, v) in adjacency_list:
        original_graph[k] = original_graph.get(k,[]) + v[:]
         
    def iterate_cgc_incremental_build(cgc):
        # incremental build of cycle-graph-closed given cycle,graph
        # this returns all possible incremental step
        (cycle, graph, over) = cgc
        if (over == True):
            return [(cycle,graph,True)]
        queue = cycle[-1]
        nnode_list = graph.get(queue,[])
        if (not nnode_list):
            # if there is no option, return same, no update, it's over
            return [(cycle,graph,True)]
        elif len(nnode_list) == 1:
            # if there is only one option do not copy struct
            # add the single node option and update graph
            nnode = nnode_list[0]
            graph.pop(queue)
            cycle.append(nnode)
            closed = (cycle[0] == cycle[-1])
            return [(cycle,graph,closed)]
        else:
            # there are multiple options from here...
            cg = []
            for nnode in nnode_list:
                agraph = copy.deepcopy(graph)
                annode_list = agraph.get(queue,[])
                annode_list.remove(nnode)
                acycle = cycle[:]
                acycle.append(nnode)
                closed = (cycle[0] == cycle[-1])
                cg.append((acycle,agraph,closed))
            return cg
        
    def inner_cycles(node,graph):
        '''
        return all possible cycles from node-node with its update unvisited edges
        graph
        '''
        # init the cycles with a single node
        lcgc = [(node,copy.deepcopy(graph),False)]
        while True:
            closed = True
            update_lcgc = []
            for cgc in lcgc:
                update_lcgc += iterate_cgc_incremental_build(cgc)
            for cgc in update_lcgc:
                closed &= cgc[2]
            if closed == True:
                # hourray, all cycles from node have been built
                return map(lambda e: (e[0], e[1]), update_lcgc)
            else:
                lcgc = update_lcgc
        

    def expand_cycles(cycle,graph):
        # search for all expandable node given graph
        lcg = []
        for n,v in enumerate(cycle):
            if v in graph:
                prefix = cycle[:n]
                suffix = cycle[n+1:]
#                print 'cycle',cycle,'expand',v,'id',n
#                print 'prefix',prefix
#                print 'icycle',icycle
#                print 'suffix',suffix
#                print '--------------'
                lcg +=  map(lambda (c,g) : (prefix+c+suffix,g), inner_cycles([v],graph))
        return lcg

    init_cycle = [original_graph.keys()[0]]
    init_graph = copy.deepcopy(original_graph)
    candidate = expand_cycles(init_cycle, init_graph)
    solution = []
    
    while candidate:
#        print 'candidate',candidate
#        print 'len',len(candidate)
        for (c,g) in candidate[:]:
            candidate.remove( (c,g) )
            if (not g):
                # c is a cycle with an empty associated graph 
                # e.g. it is a solution
                if c[0]==c[-1]:
                    solution.append(c)
            else:
                # graph is non empty let's expand another subcycles
                candidate += expand_cycles(c,g)
    
    return solution    


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

def in_and_out_degree(adjacency_list):
    '''
    return the in and out degree lists for a given graph's adjacency list
    '''
    ind = {}
    outd = {}
    for (k, v) in adjacency_list:
        outd[k] = len(v)
        for kk in v:
            ind[kk] = ind.get(kk,0)+1
    return (ind,outd)

def nearly_balanced(adjacency_list):
    '''
    return edge that will balance perfectly 
    given graph assumed to be nearly balanced
    Input : nearly balanced graph
    Output :
    balancing_edge
    '''
    (ind,outd) = in_and_out_degree(adjacency_list)
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

def genomes_reconstruction_from_pair(k, d, pairs):
    '''
    Solve the String Reconstruction from Read-Pairs Problem.
    Input: Integers k and d followed by a collection 
    of paired k-mers PairedReads.
    Output: A string Text with (k, d)-mer composition 
    equal to PairedReads.
    '''
    g = debruijn_from_pair(d,pairs)
    e = nearly_balanced(g)
    cycles = all_eularian_cycles(g + [( e[0], [e[1]] )])
    #locate precisely the edge e transition within the cycle 
    # to unwrap a path candidate
    path = []
    for c in cycles:
        for i,v in enumerate(c[:-1]):
            if (v == e[0]) and (c[i+1] == e[1]):
                    path.append(c[i+1:]+c[1:i+1])
                    break
    # filter solution as a reconstructable genome from path
    solution = set()
    for p in path:
        s = pair_genome_path(k,d,p)
        if s:
            solution.add(s)
    return list(solution)

def pair_genome_path(k,d,pairs):
    '''    
    Implement StringSpelledByGappedPatterns.
    Input: An integer d followed by a sequence of (k, d)-mers 
    (a1|b1), … , (an|bn) such that 
    Suffix(ai|bi) = Prefix(ai+1|bi+1) for 1 ≤ i ≤ n-1.
    Output: A string Text of length k + d + k + n - 1 such that 
    the i-th (k, d)-mer in Text is equal to
    (ai|bi)  for 1 ≤ i ≤ n (if such a string exists)
    '''
    # d is the gap size
    # assumes that the number pairs is at least the gap size d
    # otherwise, it is impossible to reconstruct a contiguous string.
    assert len(pairs) >= d
    prefix = genome_path(zip(*pairs)[0])
    suffix = genome_path(zip(*pairs)[1])
#    print 'prefix',prefix+'x'*(k+d)
#    print 'suffix','x'*(k+d)+suffix
    # check if prefix and suffix matches  
    if prefix[k+d:] == suffix[:-k-d]:
        return prefix + suffix[-k-d:] 
    else:
        # no matching genome found from pairs-path
        return None

def debruijn_from_pair(d,pairs):
    '''
    Construct the de Bruijn graph from a set of pairs with gap d.
    Input: A collection of k-mers pair tupples.
    Output: The adjacency list of the de Bruijn graph DeBruijn.
    '''
    g = []
    # build a prefixing pattern dict
    dprefix = {}
    for (e,f) in pairs:
        prefix = (e[:-1],f[:-1])
        suffix = (e[1:],f[1:])
        dprefix.setdefault(prefix, []).append(suffix)
    # build lexicographically sorted adjacency list
    for k in sorted(dprefix.keys()):
        g.append( (k,sorted(dprefix[k])) )
    return g


def maximal_non_branching_paths(adjacency_list):
    ''' 
    Implement MaximalNonBranchingPaths.
    Input: The adjacency list of a graph whose nodes are integers.
    Output: The collection of all maximal nonbranching paths in this graph.    
    '''
    dadj = {}
    for (k, v) in adjacency_list:
        dadj[k] = dadj.get(k,[]) + v[:]
    degree = in_and_out_degree(adjacency_list)
    paths = []
    def is_one_in_one_out(vertex):
        deg_in = degree[0].get(vertex,0)
        deg_out = degree[1].get(vertex,0)
        return (deg_in == 1) and (deg_out == 1)
    def visited(vertex):
        # linear time visited-vertex implementation
        # fixme : use dict to get constant time
        for path in paths:
            if vertex in path:
                return True
        return False
    def isolated_cycle(vertex):
        '''
        return isolated cycle including vertex v if any
        '''
        cycle = [vertex]
        while is_one_in_one_out(cycle[-1]):
            cycle.append(dadj[cycle[-1]][0])
            if cycle[0]==cycle[-1]:
                return cycle
        return None
    def non_branching_path(edge):
        '''
        return the non-branching path starting with edge edge
        '''
        branch = edge[:]
        while is_one_in_one_out(branch[-1]):     
            branch.append(dadj[branch[-1]][0])
        return branch
    for (v,e) in dadj.iteritems():
#        # cut-off optimization, skip v node if already in path list
#        if visited(v) : 
#            continue
        deg_in = degree[0].get(v,0)
        deg_out = degree[1].get(v,0)
        if (deg_in == 1 and deg_out == 1):
            # vertex v is 1-in-1-out node
            # could be part of a new isolated cycle, check this...
            if not visited(v):
                cycle = isolated_cycle(v)
                if cycle:
                    paths.append(cycle)
        elif (deg_out>0):
            # explore vertex v outgoing branches
            for w in e:
                paths.append(non_branching_path([v,w]))
    return paths

def contigs_from_reads(kmers):
    '''
    Generate the contigs from a collection of reads (with imperfect coverage).
    Input: A collection of k-mers Patterns. 
    Output: All contigs in DeBruijn(Patterns).
    '''
    g = debruijn_from_kmer(kmers)
    m = maximal_non_branching_paths(g)
    return sorted(map(genome_path,m))


assert contigs_from_reads(['ATG','ATG','TGT','TGG','CAT','GGA','GAT','AGA']) == ['AGA', 'ATG', 'ATG', 'CAT', 'GAT', 'TGGA', 'TGT']
assert maximal_non_branching_paths([(1,[2]),(2,[3]),(3,[4,5]),(6,[7]),(7,[6])]) == [[1, 2, 3], [3, 4], [3, 5], [6, 7, 6]]
assert genomes_reconstruction_from_pair(4, 2, [('GAGA','TTGA'),('TCGT','GATG'),('CGTG','ATGT'),('TGGT','TGAG'),('GTGA','TGTT'),('GTGG','GTGA'),('TGAG','GTTG'),('GGTC','GAGA'),('GTCG','AGAT')]) == ['GTGGTCGTGAGATGTTGA']     
assert pair_genome_path(4,2,[('GACC','GCGC'),('ACCG','CGCC'),('CCGA','GCCG'),('CGAG','CCGG'),('GAGC','CGGA')]) == 'GACCGAGCGCCGGA'
assert pair_composition(3,2,'TAATGCCATGGGATGTT') == [('AAT', 'CAT'), ('ATG', 'ATG'), ('ATG', 'ATG'), ('CAT', 'GAT'), ('CCA', 'GGA'), ('GCC', 'GGG'), ('GGG', 'GTT'), ('TAA', 'CCA'), ('TGC', 'TGG'), ('TGG', 'TGT')]
assert genome_reconstruction(['CTTA','ACCA','TACC','GGCT','GCTT','TTAC'])=='GGCTTACCA'
assert eulerian_path([(0,[2]),(1,[3]),(2,[1]),(3,[0,4]),(6,[3,7]),(7,[8]),(8,[9]),(9,[6])]) == [6, 7, 8, 9, 6, 3, 0, 2, 1, 3, 4]
assert rotate_cycle(eulerian_cycle([(0,[3]), (1,[0]), (2,[1,6]), (3,[2]), (4,[2]), (5,[4]), (6,[5,8]), (7,[9]), (8,[7]), (9,[6])]),0) == [0, 3, 2, 6, 8, 7, 9, 6, 5, 4, 2, 1, 0]
assert debruijn(4, 'AAGATTCTCTAAGA') == [('AAG', ['AGA', 'AGA']), ('AGA', ['GAT']), ('ATT', ['TTC']), ('CTA', ['TAA']), ('CTC', ['TCT']), ('GAT', ['ATT']), ('TAA', ['AAG']), ('TCT', ['CTA', 'CTC']), ('TTC', ['TCT'])]
assert universal_string_brute_force(2) == ['00110', '01100', '10011', '11001']
assert overlap(['ATGCG','GCATG','CATGC','AGGCA','GGCAT']) == [('AGGCA', 'GGCAT'), ('CATGC', 'ATGCG'), ('GCATG', 'CATGC'), ('GGCAT', 'GCATG')]
assert genome_path(['ACCGA','CCGAA','CGAAG','GAAGC','AAGCT']) == 'ACCGAAGCT'
assert composition(5, 'CAATCCAAC') == ['AATCC', 'ATCCA', 'CAATC', 'CCAAC', 'TCCAA']

#fname = 'C:/Users/ngaude/Downloads/dataset_205_5.txt'
#kmers = list(l[:-1] for l in open(fname))
#contigs = contigs_from_reads(kmers)
#with open(fname+'.out', "w") as f:
#    for c in contigs:
#        f.write(c+'\n')

#fname = 'C:/Users/ngaude/Downloads/MaximalNonBranchingPaths.txt'
#fname = 'C:/Users/ngaude/Downloads/dataset_6207_2.txt'
#ladj = [ ( int(l.split(' -> ')[0]), map(int,l.split(' -> ')[1].split(',')) ) for l in open(fname) ]
#mnbp = maximal_non_branching_paths(ladj)
#with open(fname+'.out', "w") as f:
#    for p in mnbp:
#        f.write(' -> '.join(map(str,p))+'\n')
#

#fname = 'C:/Users/ngaude/Downloads/StringReconstructionFromReadPairs.txt'
#fname = 'C:/Users/ngaude/Downloads/dataset_204_14.txt'
#lines = list(l for l in open(fname))
#k = int(lines[0].split(' ')[0])
#d = int(lines[0].split(' ')[1])
#pairs = map(lambda e: (e.split('|')[0].strip(), e.split('|')[1].strip()), lines[1:])
#genome = genomes_reconstruction_from_pair(k, d, pairs)
#with open(fname+'.out', "w") as f:
#        f.write(genome[0])

# quizz chapter 4
#
#kmers = ['AAAT','AATG','ACCC','ACGC','ATAC','ATCA','ATGC','CAAA','CACC','CATA','CATC','CCAG','CCCA','CGCT','CTCA','GCAT','GCTC','TACG','TCAC','TCAT','TGCA']
#print 'quizz',genome_reconstruction(kmers)
#

#us = universal_string_brute_force(3)
#quizz = ['1000101110','1101000111','1100011011','0101001101','0101010100','0111010001']
##quizz  = ['1100011011','0111010010','0101001101','0011101000','0111010001','1110001011']
#for c in quizz:
#    print c,c in us

   
#
#kmers = ['GCGA','CAAG','AAGA','GCCG','ACAA','AGTA','TAGG','AGTA','ACGT','AGCC','TTCG','AGTT','AGTA','CGTA','GCGC','GCGA','GGTC','GCAT','AAGC','TAGA','ACAG','TAGA','TCCT','CCCC','GCGC','ATCC','AGTA','AAGA','GCGA','CGTA']
#g = debruijn_from_kmer(kmers)
#for (e,ee) in g:
#    if ('GTA' in ee): 
#        print e,'==>',ee
#

#pairs = [('ACC', 'ATA'),('ACT', 'ATT'),('ATA', 'TGA'),('ATT', 'TGA'),('CAC', 'GAT'),('CCG', 'TAC'),('CGA', 'ACT'),('CTG', 'AGC'),('CTG', 'TTC'),('GAA', 'CTT'),('GAT', 'CTG'),('GAT', 'CTG'),('TAC', 'GAT'),('TCT', 'AAG'),('TGA', 'GCT'),('TGA', 'TCT'),('TTC', 'GAA')]
#
#pairs = [('ACC','ATA'),
#('ACT','ATT'),
#('ATA','TGA'),
#('ATT','TGA'),
#('CAC','GAT'),
#('CCG','TAC'),
#('CGA','ACT'),
#('CTG','AGC'),
#('CTG','TTC'),
#('GAA','CTT'),
#('GAT','CTG'),
#('GAT','CTG'),
#('TAC','GAT'),
#('TCT','AAG'),
#('TGA','GCT'),
#('TGA','TCT'),
#('TTC','GAA')]
#g = genomes_reconstruction_from_pair(3, 1, pairs)
#print g


##############################


#fname = 'C:/Users/ngaude/Downloads/dataset_6206_7.txt'
#lines = list(l for l in open(fname))
#d = int(lines[0])
#pairs = map(lambda e: (e.split('|')[0].strip(), e.split('|')[1].strip()), lines[1:])
#genome = pair_genome_path(d,pairs)
#with open(fname+'.out', "w") as f:
#        f.write(genome)

#p = pair_composition(3,2,'TAATGCCATGGGATGTT')
#print  '('+')('.join(map(lambda e : e[0]+'|'+e[1], p))+')'

#fname = 'C:/Users/ngaude/Downloads/dataset_203_6.txt'
#kmers = list(l[:-1] for l in open(fname))[1:]
#genome = genome_reconstruction(kmers)
#with open(fname+'.out', "w") as f:
#        f.write(genome)

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