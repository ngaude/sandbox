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

def bwmatching(s,patterns):
    """
    CODE CHALLENGE: Implement BWMATCHING.
    Input: A string BWT(Text), followed by a collection of Patterns.
    Output: A list of integers, where the i-th integer corresponds to the number of substring
    matches of the i-th member of Patterns in Text.
    """    
    def pattern_count(first_column,last_column,pattern,last_to_first):
        top = 0
        bottom = len(last_column) - 1
        while top <= bottom:
            if pattern:
                symbol = pattern[-1]
                pattern = pattern[:-1]
                if symbol in last_column[top:bottom+1]:
                    top_index = last_column.find(symbol,top,bottom+1)
                    bottom_index = last_column.rfind(symbol,top,bottom+1)
                    top = last_to_first[top_index]
                    bottom = last_to_first[bottom_index]
                else:
                    return 0
            else:
                return bottom - top + 1
        return 0    
    l = len(s)
    # produce a list tuple (char,index) for the last column
    last_char_rank = [(s[i],i) for i in range(l)]
    # produce the list tuple (char,rank) for the first column
    first_char_rank = sorted(last_char_rank)
    # build the last_to_first conversion array
    
    first_to_last = [ i for (c,i) in first_char_rank]
    last_to_first = [None]*l
    for first,last in enumerate(first_to_last):
        last_to_first[last] = first
    first_column = sorted(s)
    last_column = s
    
#    for i in range(l):
#        r = str(first_column[i])+('*'*(l-2))+str(last_column[i])
#        rr = str(last_column[first_to_last[i]])+('*'*(l-2))+str(first_column[last_to_first[i]])
#        assert rr == r
    
    return [pattern_count(first_column,last_column,pattern,last_to_first) for pattern in patterns]

assert bwmatching(bwt('panamabananas$'),['pan','ana']) == [1,3]
assert bwmatching('TCCTCTATGAGATCCTATTCTATGAAACCTTCA$GACCAAAATTCTCCGGC',['CCT', 'CAC', 'GAG', 'CAG', 'ATC']) == [2, 1, 1, 0, 1]

    
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

#fname = 'C:/Users/ngaude/Downloads/dataset_300_8.txt'
#with open(fname, "r") as f:
#    text = f.read().strip().split('\n')
#    s = text[0]
#    p = text[1].split(' ')
#with open(fname+'.out', "w") as f:
#    f.write(' '.join(map(str,bwmatching(s,p))))
