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

def ibwt(s):
    """
    Inverse Burrows-Wheeler Transform Problem: 
    Reconstruct a string from its Burrows-Wheeler transform.
    Input: A string Transform (with a single "$" symbol).
    Output: The string Text such that BWT(Text) = Transform.
    """
    l = len(s)
    
    def char_rank(i):
        d[s[i]] = d.get(s[i],0) + 1
        return d[s[i]]
    d = {}
    # produce a list tuple (char,rank) for the last column
    last_char_rank = [(s[i],char_rank(i),i) for i in range(l)]
    d = {}
    # produce the list tuple (char,rank) for the first column
    first_char_rank = sorted(last_char_rank)
        
#    for i in range(l):
#        r = str(first_char_rank[i])+('*'*(l-2))+str(last_char_rank[i])
#        print r
    
    i = 0
    decoded = ''
    for j in range(l):
        i = first_char_rank[i][2]
        decoded += first_char_rank[i][0]
    return decoded
    
   
assert ibwt('ard$rcaaaabb') == 'abracadabra$'
assert ibwt('TTCCTAACG$A') == 'TACATCACGT$'


"""
    BWMATCHING(FirstColumn, LastColumn, Pattern, LastToFirst)
        top ← 0
        bottom ← |LastColumn| − 1
        while top ≤ bottom
            if Pattern is nonempty
                symbol ← last letter in Pattern
                remove last letter from Pattern
                if positions from top to bottom in LastColumn contain an occurrence of symbol
                    topIndex ← first position of symbol among positions from top to bottom in LastColumn
                    bottomIndex ← last position of symbol among positions from top to bottom in LastColumn
                    top ← LastToFirst(topIndex)
                    bottom ← LastToFirst(bottomIndex)
                else
                    return 0
            else
                return bottom − top + 1
"""
 
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

#fname = 'C:/Users/ngaude/Downloads/dataset_299_10.txt'
#with open(fname, "r") as f:
#    text = f.read().strip()
#with open(fname+'.out', "w") as f:
#    f.write(ibwt(text))           
        
