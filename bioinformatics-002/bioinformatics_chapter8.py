# -*- coding: utf-8 -*-
"""
Created on Wed Apr 01 21:41:45 2015

@author: ngaude
"""

def suffix_array(s):
    """
    Suffix Array Construction Problem: Construct the suffix array of a string.
    Input: A string Text.
    Output: SuffixArray(Text).
    """
    # elegant by uses too much memory 
    # return sorted(range(len(s)), key=lambda i: s[i:])
    
    # no memory issue, but still not time loglinear because of string copy
    # return sorted(range(len(s)), cmp=lambda i,j: cmp(s[i:],s[j:]))
    
    # no memory issue, no string copy issue, but python bytecode is still slow
    l = len(s)
    def compare(i,j):        
        while i<l and j<l:
            if s[i]>s[j]:
                return 1
            elif s[i]<s[j]:
                return -1
            i +=1
            j +=1
        return 0
    return sorted(range(len(s)), cmp=compare)
    
assert suffix_array('AACGATAGCGGTAGA$') == [15, 14, 0, 1, 12, 6, 4, 2, 8, 13, 3, 7, 9, 10, 11, 5]

def bwt(s):
    """
    Burrows-Wheeler Transform Construction Problem: 
    Construct the Burrows-Wheeler transform of a string.
    Input: A string Text.
    Output: BWT(Text).
    """
    return ''.join([s[(i-1) % len(s)] for i in suffix_array(s)])

assert bwt('GCGTGCCTGGTCA$') == 'ACTGGCT$TGCGGC'

    
########################

#fname = 'C:/Users/ngaude/Downloads/dataset_310_2.txt'
#with open(fname, "r") as f:
#    text = f.read().strip()
#s = ', '.join(map(str,suffix_array(text)))
#with open(fname+'.out', "w") as f:
#    f.write(s)
    
#fname = 'C:/Users/ngaude/Downloads/dataset_297_4.txt'
#with open(fname, "r") as f:
#    text = f.read().strip()
#with open(fname+'.out', "w") as f:
#    f.write(bwt(text))

#import time
#duration = -time.time()   
#fname = 'C:/Users/ngaude/Downloads/E-coli.txt'
#with open(fname, "r") as f:
#    text = f.read().strip()
#with open(fname+'.out', "w") as f:
#    f.write(bwt(text))
#duration += time.time()
#print duration

#fname = 'C:/Users/ngaude/Downloads/E-coli.txt.out'
#with open(fname, "r") as f:
#    text = f.read().strip()
#
#rlecount = 0    
#rep = 0
#seq = ''
#maxrep = 0
#for c in text:
#    if seq == c:
#        rep +=1
#    else:
#        maxrep = max(rep,maxrep)
#        if rep>9:
#            rlecount +=1
#        rep = 1
#        seq = c
#print maxrep,rlecount
            
        