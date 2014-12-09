# -*- coding: utf-8 -*-
"""
Created on Mon Dec 08 22:45:00 2014

@author: ngaude
"""

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

def spell_path(path):
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

def universal_string(k):
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

u = universal_string(3)

assert overlap(['ATGCG','GCATG','CATGC','AGGCA','GGCAT']) == [('AGGCA', 'GGCAT'), ('CATGC', 'ATGCG'), ('GCATG', 'CATGC'), ('GGCAT', 'GCATG')]
assert spell_path(['ACCGA','CCGAA','CGAAG','GAAGC','AAGCT']) == 'ACCGAAGCT'
assert composition(5, 'CAATCCAAC') == ['AATCC', 'ATCCA', 'CAATC', 'CCAAC', 'TCCAA']


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
#o = spell_path(p)
#print o

#fname = 'C:/Users/ngaude/Downloads/dataset_197_3.txt'
#(k, text)  = (l[:-1] for l in open(fname))
#o = '\n'.join(composition(int(k),text))
#with open(fname+'.out', "w") as f: f.write(o)