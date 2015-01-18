# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 13:42:39 2015

@author: ngaude
"""

import numpy as np

def reversal(p, i, k):
    prefix = p[:i]
    suffix = p[k+1:]
    root = p[i:k+1]
    return prefix + map(lambda x: -x, root[::-1]) + suffix

def greedy_sorting(p):
    '''
    CODE CHALLENGE: Implement GREEDYSORTING.
    Input: A permutation P.
    Output: The sequence of permutations corresponding to 
    applying GREEDYSORTING to P, ending with
    the identity permutation.
    '''
    s = []
    for i in range(1,len(p)+1):
        try:
            k = p.index(-i)
            p = reversal(p,i-1,k)
            s.append(p)
        except ValueError:
            k = p.index(i)
            if (k != i-1):
                p = reversal(p,i-1,k)
                s.append(p)
                p = p[:]
                p[i-1] = -p[i-1]
                s.append(p)
    return s

def permutation_list_to_str(p):
    def str_val(i):
        if (i>0):
            return '+'+str(i)
        else:
            return str(i)
    return '(' + ' '.join(map(str_val,p)) + ')'

def permutation_str_to_list(str_p):
    p = map(int,str_p.strip()[1:-1].split(' '))
    return p
        
def format_sequence(s):
    fs = []
    for p in s:
        str_p = permutation_list_to_str(p)
        fs.append(str_p)
    return fs
    

def number_of_breakpoints(p):
        '''
        CODE CHALLENGE: Solve the Number of Breakpoints Problem.
        
        Number of Breakpoints Problem: Find the number of breakpoints in a permutation.
        Input: A permutation.
        Output: The number of breakpoints in this permutation.        
        '''
        adj = 0
        p = [0,] + p + [len(p)+1]
        for i in range(0,len(p)-1):
            if (p[i+1]==p[i]+1):
                adj += 1
        return len(p) - 1 - adj

def permutation_irreductible(p):
    min_bk = number_of_breakpoints(p)
    for i in range(0,len(p)):
        for j in range(i,len(p)):
            bk = number_of_breakpoints(reversal(p,i,j))
            print min_bk,bk
            if bk < min_bk:
                return False
    return True
    

def chromosome_to_cycle(p):
    '''
    CODE CHALLENGE: Implement ChromosomeToCycle.
    Input: A chromosome Chromosome containing n synteny blocks.
    Output: The sequence Nodes of integers between 1 and 2n resulting 
    from applying ChromosomeToCycle to Chromosome.
    '''
    nodes = []
    
    for i in p:
        if (i>0):
            nodes.append(2*i-1)
            nodes.append(2*i)
        else:
            nodes.append(-2*i)
            nodes.append(-2*i-1)
    return nodes

def cycle_to_chromosome(nodes):
    '''
    CODE CHALLENGE: Implement CycleToChromosome.
    Input: A sequence Nodes of integers between 1 and 2n.
    Output: The chromosome Chromosome containing n synteny blocks resulting 
    from applying CycleToChromosome to Nodes.
    '''
    p = []
    for j in range(0,len(nodes)/2):
        if nodes[2*j] < nodes[2*j+1]:
            s = j+1
        else:
            s = -(j+1)
        p.append(s)
    return p


def genome_str_to_list(genome):
    lp = []
    for p in genome.split('(')[1:]:
        p = permutation_str_to_list( '(' + p )
        lp.append(p)
    return lp

def colored_edges(genome):
    '''
    CODE CHALLENGE: Implement ColoredEdges.
    Input: A genome P.
    Output: The collection of colored edges in the genome graph of P 
    in the form (x, y).
    '''
    g = []
    for p in genome:
        s = chromosome_to_cycle(p)
        for j in range(len(s)/2):
            head = 1+2*j
            tail = (2+2*j) % len(s)
            e = (s[head],s[tail])
            g.append(e)
    return g
    
def graph_to_genome(g):
    '''
    CODE CHALLENGE: Implement GraphToGenome.
    Input: The colored edges ColoredEdges of a genome graph.
    Output: The genome P corresponding to this genome graph.
    '''
    
    genome = []
    visit = []
    adj = np.zeros(len(g)*2, dtype = np.int)
    for t in g:
        adj[t[0]-1] = t[1]-1
        adj[t[1]-1] = t[0]-1
    
    for t in g:
        orig = t[0]
        if orig in visit:
            continue
        visit.append(orig)
        if (orig%2 == 0):
            closing = orig-1
        else:
            closing = orig+1
        p = []
        i = 0
        while(True):
            if (orig%2 == 0):
                p.append(orig/2)
            else:
                p.append(-(orig+1)/2)
            dest = adj[orig-1]+1
            i = i + 1
            if (i>100):
                assert False
                return
            visit.append(dest)
            if (dest == closing):
                genome.append(p)
                break
            if (dest%2 == 0):
                orig = dest -1
            else:
                orig = dest + 1
            assert orig > 0
            visit.append(orig)
    return genome
  
def colored_edges_cycles(blue, red):
    '''    
    returns all alternating red-blue-edge cycles
    '''
    cycles = []
    size = len(blue)+len(red) 
    adj = np.zeros(shape = (size,2), dtype = np.int)
    visited = np.zeros(shape = size, dtype = np.bool)
    for e in blue:
        adj[e[0]-1,0] = e[1]-1
        adj[e[1]-1,0] = e[0]-1
    for e in red:
        adj[e[0]-1,1] = e[1]-1
        adj[e[1]-1,1] = e[0]-1
    
    for node in range(size):
        if not visited[node]:
            visited[node] = True
            head = node
            cycle = [head+1]
            # arbitrary we start with a blue edge
            color = 0
            while (True):
                node = adj[node,color]
                if (node == head):
                    # cycle found, we got back to the visited head node, 
                    # just break
                    cycles.append(cycle)
                    break
                cycle.append(node+1)
                visited[node] = True
                color = (color+1) % 2
    return cycles

def two_break_distance(P, Q):
    '''
    CODE CHALLENGE: Solve the 2-Break Distance Problem.
    Input: Genomes P and Q.
    Output: The 2-break distance d(P, Q).
    '''
    blue = colored_edges(P)
    red = colored_edges(Q)

    assert len(blue) == len(red)
    assert len(blue)+len(red) == max([element for tupl in blue+red for element in tupl])
    
    size = len(blue)+len(red) 
    
    l = colored_edges_cycles(blue,red)
    return size/2 - len(l)

def two_break_on_genome_graph(g,i,j,k,l):
    '''
    CODE CHALLENGE: Implement 2-BreakOnGenomeGraph.
    Input: The colored edges of a genome graph GenomeGraph, 
    followed by indices i, j, k, and l.
    Output: The colored edges of the genome graph resulting from applying 
    the 2-break operation 2-BreakOnGenomeGraph(GenomeGraph, i, j, k, l).
    '''
    bg = []
    # equivalent and more elegant but not accepted by the grader ...
#    d = {(i,j):(i,k), (j,i):(j,l), (k,l):(k,i), (l,k):(l,j)}    
#    for t in g:
#        if (t in d):
#            bg.append(d[t])
#        else:
#            bg.append(t)
    
    # so do it this way
    rem = ((i,j), (j,i), (k,l), (l,k))
    bg = [ t for t in g if t not in rem]
    bg.append((i,k))
    bg.append((j,l))
    
    return bg

def two_break_on_genome(genome,i,j,k,l):
    '''
    CODE CHALLENGE: Implement 2-BreakOnGenome.
    Input: A genome P, followed by indices i, i', j, and j'.
    Output: The genome P' resulting from applying the 2-break operation
    2-BreakOnGenomeGraph(GenomeGraph, i, i′, j, j′).
    '''
    g = colored_edges(genome)
    g = two_break_on_genome_graph(g,i,j,k,l)
    genome = graph_to_genome(g)
    return genome


def two_break_sorting(P,Q):
    '''
    CODE CHALLENGE: Solve the 2-Break Sorting Problem.     
    2-Break Sorting Problem: Find a shortest transformation 
    of one genome into another via 2-breaks.
    Input: Two genomes with circular chromosomes on the same 
    set of synteny blocks.
    Output: The collection of genomes resulting from applying 
    a shortest sequence of 2-breaks transforming one genome into the other.
    '''
    red = colored_edges(Q)
    path = [P]
    while two_break_distance(P,Q) > 0:
        cycles = colored_edges_cycles(colored_edges(P),red)
        for c in cycles:
            if len(c) >= 4:
                P = two_break_on_genome(P,c[0],c[1],c[3],c[2])
                path.append(P)
                break          
    return path

k = 3
a = 'AAACTCATC'
b = 'TTTCAAATC'

def shared_kmers(k,a,b):
    '''
    CODE CHALLENGE: Solve the Shared k-mers Problem.
    Shared k-mers Problem: Given two strings, find all their shared k-mers.
    Input: An integer k and two strings.
    Output: All k-mers shared by these strings, 
    in the form of ordered pairs (x, y).
    '''    
    def reverse_complement(pattern):
        rev = {'A':'T', 'T':'A', 'G':'C', 'C':'G'}
        reverse = map(lambda c: rev[c], pattern[::-1])
        return ''.join(reverse)
    
    def kmers_dict(k, text):
        ''' 
        Solve the String Composition Problem.
        Input: An integer k and a string Text.
        Output: returns a k-mers:[positions,] dictionnary
        '''
        kmers = {}
        for i in range(len(text) - k + 1):
            kmer = text[i:i+k]
            kmers[kmer] = kmers.setdefault(kmer,[]) + [i]
            kmers[reverse_complement(kmer)] = kmers[kmer]
        return kmers
    
    shared = []
        
    bkmers = kmers_dict(k,b)
    for i in range(len(a) - k + 1):
        akmer = a[i:i+k]
        if akmer in bkmers:
            shared += [(i,j) for j in bkmers[akmer]]
    
    return sorted(shared)
        

fname = 'C:/Users/ngaude/Downloads/dataset_289_5.txt'
lines = list(l for l in open(fname))
k = int(lines[0])
a = lines[1].strip()
b = lines[2].strip()
with open(fname+'.out', "w") as f:
        f.write('\n'.join(map(str,shared_kmers(k,a,b))))


k = 3
a = 'TGGCCTGCACGGTAG'
b = 'GGACCTACAAATGGC'


assert len(greedy_sorting([-3,4,1,5,-2])) == 7
assert number_of_breakpoints([3, 4, 5, -12, -8, -7, -6, 1, 2, 10, 9, -11, 13, 14]) == 8
assert chromosome_to_cycle([1,-2,-3,4]) == [1, 2, 4, 3, 6, 5, 7, 8]
assert cycle_to_chromosome([1, 2, 4, 3, 6, 5, 7, 8]) == [1,-2,-3,4]
assert colored_edges([[1, -2, -3], [4, 5, -6]]) == [(2, 4), (3, 6), (5, 1), (8, 9), (10, 12), (11, 7)]
assert graph_to_genome([(2, 4), (3, 6), (5, 1), (7, 9), (10, 12), (11, 8)]) == [[1, -2, -3], [-4, 5, -6]]
assert two_break_distance([[1, 2, 3, 4, 5, 6]],[[1, -3, -6, -5], [2, -4]]) == 3
assert two_break_on_genome_graph([(2, 4), (3, 8), (7, 5), (6, 1)],1,6,3,8) == [(2, 4), (7, 5), (1, 3), (6, 8)]
assert two_break_on_genome([[1,-2,-4,3]],1, 6, 3, 8) == [[1, -2], [-4, 3]]
assert [two_break_distance(p,[[1, 2, -4, -3]]) for p in two_break_sorting([[1, -2, -3, 4]],[[1, 2, -4, -3]])] == range(4)[::-1]
assert shared_kmers(3,'AAACTCATC','TTTCAAATC') == [(0, 0), (0, 4), (4, 2), (6, 6)]

#
#P = '(+9 -8 +12 +7 +1 -14 +13 +3 -5 -11 +6 -2 +10 -4)'
#Q = '(-11 +8 -10 -2 +3 +4 +13 +6 +12 +9 +5 +7 -14 -1)'
#
#P = '(+4 +1 -7 +10 -8 -2 +5 +6 -11 +3 +12 +9)'
#Q = '(-8 -7 +4 +6 -1 +5 -10 -11 +9 +12 +2 -3)'
#P = genome_str_to_list(P)
#Q = genome_str_to_list(Q)
#path = two_break_sorting(P,Q)
#print '\n'.join([''.join(format_sequence(p)) for p in path])

#genome = '(+1 -2 -3 +4 -5 +6 +7 -8 +9 +10 +11 -12 +13 -14 -15 +16 -17 -18 +19 -20 +21 +22 -23 -24 +25 -26 -27 -28 -29 -30 +31 -32 +33 -34 -35 +36 +37 -38 -39 +40 +41 +42 -43 +44 +45 -46 +47 +48 +49 +50 +51 -52 +53 +54 -55 -56 -57 -58 +59 -60 -61 -62 +63)'
#i,j,k,l = 8, 10, 116, 113
#genome = genome_str_to_list(genome)
#genome = two_break_on_genome(genome,i,j,k,l)
#fs = format_sequence(genome)
#print ''.join(fs)

#g = [(1, 3), (4, 5), (6, 7), (8, 9), (10, 12), (11, 14), (13, 15), (16, 18), (17, 20), (19, 21), (22, 23), (24, 26), (25, 27), (28, 29), (30, 32), (31, 33), (34, 35), (36, 37), (38, 40), (39, 42), (41, 43), (44, 46), (45, 48), (47, 49), (50, 51), (52, 54), (53, 55), (56, 57), (58, 59), (60, 61), (62, 63), (64, 65), (66, 67), (68, 70), (69, 72), (71, 74), (73, 75), (76, 77), (78, 79), (80, 81), (82, 83), (84, 86), (85, 88), (87, 89), (90, 92), (91, 93), (94, 96), (95, 97), (98, 100), (99, 101), (102, 103), (104, 106), (105, 108), (107, 110), (109, 112), (111, 114), (113, 115), (116, 117), (118, 120), (119, 122), (121, 123), (124, 126), (125, 127), (128, 129), (130, 131), (132, 133), (134, 136), (135, 2)]
#(i,j,k,l) = (109, 112, 64, 65)
#print two_break_on_genome_graph(g,i,j,k,l)

#fname = 'C:/Users/ngaude/Downloads/dataset_288_4.txt'
#lines = list(l for l in open(fname))
#P = lines[0]
#Q = lines[1]
#P = genome_str_to_list(P)
#Q = genome_str_to_list(Q)      
#print two_break_distance(P,Q) 

#fname = 'C:/Users/ngaude/Downloads/2_break.txt'
#lines = list(l for l in open(fname))
#P = lines[0]
#Q = lines[1]
#P = genome_str_to_list(P)
#Q = genome_str_to_list(Q)      
#print two_break_distance(P,Q)

#g = [(1, 4), (3, 6), (5, 7), (8, 9), (10, 11), (12, 14), (13, 16), (15, 18), (17, 20), (19, 22), (21, 23), (24, 25), (26, 28), (27, 30), (29, 32), (31, 34), (33, 35), (36, 38), (37, 40), (39, 41), (42, 44), (43, 46), (45, 48), (47, 49), (50, 2), (52, 53), (54, 56), (55, 57), (58, 60), (59, 61), (62, 63), (64, 65), (66, 67), (68, 69), (70, 72), (71, 74), (73, 76), (75, 77), (78, 79), (80, 82), (81, 83), (84, 85), (86, 88), (87, 90), (89, 51), (91, 93), (94, 96), (95, 97), (98, 100), (99, 101), (102, 103), (104, 106), (105, 108), (107, 109), (110, 112), (111, 114), (113, 115), (116, 117), (118, 120), (119, 122), (121, 124), (123, 126), (125, 128), (127, 130), (129, 92), (131, 134), (133, 136), (135, 138), (137, 140), (139, 142), (141, 144), (143, 145), (146, 148), (147, 149), (150, 152), (151, 153), (154, 155), (156, 158), (157, 160), (159, 161), (162, 163), (164, 166), (165, 168), (167, 169), (170, 172), (171, 174), (173, 175), (176, 178), (177, 179), (180, 182), (181, 184), (183, 185), (186, 188), (187, 132), (189, 192), (191, 193), (194, 196), (195, 197), (198, 199), (200, 201), (202, 203), (204, 206), (205, 208), (207, 210), (209, 212), (211, 214), (213, 215), (216, 217), (218, 220), (219, 222), (221, 224), (223, 226), (225, 227), (228, 229), (230, 232), (231, 233), (234, 190), (235, 237), (238, 239), (240, 242), (241, 244), (243, 246), (245, 248), (247, 250), (249, 252), (251, 253), (254, 255), (256, 258), (257, 260), (259, 262), (261, 263), (264, 265), (266, 268), (267, 269), (270, 271), (272, 273), (274, 275), (276, 277), (278, 280), (279, 281), (282, 284), (283, 286), (285, 287), (288, 289), (290, 292), (291, 293), (294, 236), (296, 297), (298, 300), (299, 302), (301, 304), (303, 306), (305, 307), (308, 309), (310, 311), (312, 314), (313, 315), (316, 318), (317, 319), (320, 321), (322, 324), (323, 325), (326, 327), (328, 329), (330, 332), (331, 334), (333, 336), (335, 337), (338, 340), (339, 342), (341, 344), (343, 346), (345, 348), (347, 349), (350, 352), (351, 295), (354, 356), (355, 357), (358, 360), (359, 361), (362, 363), (364, 366), (365, 368), (367, 370), (369, 371), (372, 374), (373, 375), (376, 377), (378, 379), (380, 382), (381, 384), (383, 385), (386, 388), (387, 389), (390, 391), (392, 393), (394, 353), (396, 397), (398, 399), (400, 402), (401, 404), (403, 405), (406, 407), (408, 409), (410, 411), (412, 414), (413, 416), (415, 418), (417, 419), (420, 421), (422, 424), (423, 425), (426, 428), (427, 429), (430, 431), (432, 434), (433, 435), (436, 437), (438, 440), (439, 442), (441, 443), (444, 446), (445, 447), (448, 449), (450, 452), (451, 453), (454, 395)]
#genome = graph_to_genome(g)
#print ''.join(format_sequence(genome))

#genome = '(-1 +2 -3 -4 -5 +6 -7 +8 +9 +10 +11 -12 +13 -14 -15 +16 +17 -18 -19 +20 +21 +22 +23 -24 +25 -26 -27 -28 +29 -30 -31)(-32 +33 -34 -35 -36 +37 -38 +39 -40 +41 +42 -43 +44 -45 -46 -47 +48 -49 +50 -51 +52 -53 +54)(-55 -56 +57 +58 +59 +60 -61 -62 +63 -64 -65 -66 -67 +68 -69 +70 -71 +72 -73 +74 -75 +76 -77 +78 +79 +80 -81)(-82 -83 +84 -85 -86 +87 -88 -89 +90 +91 +92 +93 -94 -95 +96 -97 -98 +99 -100 -101 +102 +103 +104 +105 +106 -107 -108 +109 +110 -111)(-112 +113 +114 -115 -116 +117 +118 -119 -120 -121 +122 +123 -124 -125 +126 -127 +128 +129 -130 -131 -132 +133 -134 -135)(-136 -137 +138 -139 -140 -141 +142 -143 -144 -145 -146 -147 +148 +149 +150 +151 +152 +153 -154 -155 -156 -157 -158 +159 -160 -161)(+162 +163 -164 -165 -166 +167 +168 +169 -170 -171 -172 -173 +174 +175 +176 +177 +178 -179 -180 +181 +182 -183 -184)(+185 -186 -187 +188 +189 +190 -191 -192 +193 -194 -195 +196 -197 +198 +199 -200 -201 -202 -203 +204 -205 +206)'
#genome = genome_str_to_list(genome)
#col = colored_edges(genome)
#print '(' + ', '.join(map(str,col)) + ')'

#s = '(2 1 3 4 6 5 8 7 9 10 12 11 14 13 15 16 18 17 20 19 21 22 24 23 25 26 28 27 29 30 31 32 33 34 36 35 37 38 40 39 41 42 43 44 45 46 48 47 49 50 52 51 54 53 55 56 57 58 59 60 62 61 63 64 65 66 68 67 69 70 72 71 74 73 75 76 78 77 79 80 82 81 84 83 86 85 87 88 90 89 91 92 93 94 96 95 97 98 100 99 102 101 104 103 106 105 107 108 109 110 111 112 113 114 115 116 117 118 120 119 122 121 123 124 126 125 128 127 130 129 131 132 134 133)'
#s = permutation_str_to_list(s)
#p = cycle_to_chromosome(s)
#print permutation_list_to_str(p)

#fname = 'C:/Users/ngaude/Downloads/number_of_breaks.txt'
#with open(fname, "r") as f:
#    p = f.read()
#p = permutation_str_to_list(p)
#print number_of_breakpoints(p)
#
#fname = 'C:/Users/ngaude/Downloads/dataset_287_4.txt'
#with open(fname, "r") as f:
#    p = f.read()
#p = permutation_str_to_list(p)
#print number_of_breakpoints(p)



#p = '(100 '+' '.join(map(str,range(1,100)))+')'
#p = permutation_str_to_list(p)
#s = greedy_sorting(p)
#fs = format_sequence(s)
#print len(fs)

#fname = 'C:/Users/ngaude/Downloads/dataset_286_3.txt'
#with open(fname, "r") as f:
#    p = f.read()
#p = p = permutation_str_to_list(p)
#s = greedy_sorting(p)
#fs = format_sequence(s)
#with open(fname+'.out', "w") as f:
#    for p in fs:
#        f.write(p+'\n')

#######################################################
# QUIZZ
#######################################################

#p  = '(+20 +7 +10 +9 +11 +13 +18 -8 -6 -14 +2 -4 -16 +15 +1 +17 +12 -5 +3 -19)'
#p = permutation_str_to_list(p)
#s = greedy_sorting(p)
#fs = format_sequence(s)
#print 'l=',len(fs)
#
#p = '(+10 +6 -8 -7 +17 -20 +18 +19 -5 -16 -11 -4 -3 -2 +13 +14 -1 +9 -12 +15)'
#p = permutation_str_to_list(p)
#print 'n=',number_of_breakpoints(p)




k = 3
a = 'TGGCCTGCACGGTAG'
b = 'GGACCTACAAATGGC'
print len(shared_kmers(k,a,b))
