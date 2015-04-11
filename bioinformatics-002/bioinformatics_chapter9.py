# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:18:11 2015

@author: ngaude
"""

from bioinformatics_chapter8 import suffix_array
from bioinformatics_chapter8 import bwt

def partial_suffix_array(s,k):
    """
    CODE CHALLENGE: Construct a partial suffix array.
    Input: A string Text and a positive integer K.
    Output: SuffixArrayK(Text), in the form of a list of ordered pairs (i, SuffixArray(i)) for all
    nonempty entries in the partial suffix array.
    """
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
    sa = sorted(range(len(s)), cmp=compare)
    psa = [(i,c) for (i,c) in enumerate(sa) if c % k == 0]
    return psa
    
assert partial_suffix_array('panamabananas$',5) == [(1, 5), (11, 10), (12, 0)]

def better_bw_matching(s,patterns):
    """
    CODE CHALLENGE: Implement BETTERBWMATCHING.
    Input: A string BWT(Text) followed by a collection of strings Patterns.
    Output: A list of integers, where the i-th integer corresponds to the number of substring
    matches of the i-th member of Patterns in Text.
    """
    # create count_symbol array : len = |s| * #symbol 
    def scan_symbol_count(c):
        # update current dict
        current_dict[c] = current_dict.get(c,0)+1
        # copy current dict to the current symbol array
        return current_dict.copy()
    current_dict = {}
    count_symbol = [{}] + [scan_symbol_count(c) for c in s]
    
    # create first_occurence 
    def scan_first_occurence((i,c)):
        if c not in first_occurence:
            first_occurence[c] = i
    first_occurence = {}    
    map(scan_first_occurence,enumerate(sorted(s)))

    def print_symbol_count():
        symbols = sorted(set(s))
        last_col = ' ' + s
        first_col = ' ' + ''.join(sorted(s))

        header = 'f l ##:'+' '.join(symbols)        
        
        delimiter = ''.join(['-']*len(header))
        print delimiter
        print header
        print delimiter
        for i,d in enumerate(count_symbol):
            l = first_col[i] + ' ' + last_col[i] + ' ' + format(i,'02') + ':'
            l += ' '.join(map(lambda symbol:str(d.get(symbol,0)),symbols))
            print l
        print delimiter

#    print_symbol_count()
    
    def pattern_count(pattern):
        top = 0
        bottom = len(s) - 1
        while top <= bottom:
            if pattern:
                symbol = pattern[-1]
                pattern = pattern[:-1]
                if count_symbol[bottom+1].get(symbol,0) > count_symbol[top].get(symbol,0):   
                        top = first_occurence[symbol] + count_symbol[top].get(symbol,0)
                        bottom = first_occurence[symbol] + count_symbol[bottom+1].get(symbol,0) - 1
                else:
                    return 0
            else:
                return bottom - top + 1
        return 0
    
    return [pattern_count(pattern) for pattern in patterns]
    
assert better_bw_matching('GGCGCCGC$TAGTCACACACGCCGTA',['ACC', 'CCG', 'CAG']) == [1, 2, 1]

def multiple_pattern_matching(text,patterns):
    """
    CODE CHALLENGE: Solve the Multiple Pattern Matching Problem.
    Input: A string Text followed by a collection of strings Patterns.
    Output: All starting positions in Text where a string from Patterns appears as a substring.
    """
    # to cope with not $-ending text
    if text[-1] != '$':
            text+='$'
    s = bwt(text)
    a = suffix_array(text)
    
    
    def scan_symbol_count(c):
        # update current dict
        current_dict[c] = current_dict.get(c,0)+1
        # copy current dict to the current symbol array
        return current_dict.copy()
    current_dict = {}
    count_symbol = [{}] + [scan_symbol_count(c) for c in s]
    
    # create first_occurence 
    def scan_first_occurence((i,c)):
        if c not in first_occurence:
            first_occurence[c] = i
    first_occurence = {}    
    map(scan_first_occurence,enumerate(sorted(s)))
    
    def pattern_positions(pattern):
        top = 0
        bottom = len(s) - 1
        while top <= bottom:
            if pattern:
                symbol = pattern[-1]
                pattern = pattern[:-1]
                if count_symbol[bottom+1].get(symbol,0) > count_symbol[top].get(symbol,0):   
                        top = first_occurence[symbol] + count_symbol[top].get(symbol,0)
                        bottom = first_occurence[symbol] + count_symbol[bottom+1].get(symbol,0) - 1
                else:
                    return []
            else:
                return [a[i] for i in range(top,bottom+1)]
        return []
    
    pos = []
    for pattern in patterns:
        pos += pattern_positions(pattern)
    pos.sort()
    return pos

assert multiple_pattern_matching('AATCGGGTTCAATCGGGGT',('ATCG','GGGT')) == [1, 4, 11, 15]
    
def multiple_approximate_pattern_matching(text,patterns,d):
    """
    CODE CHALLENGE: Solve the Multiple Approximate Pattern Matching Problem.
    Input: A string Text, followed by a collection of strings Patterns, and an integer d.
    Output: All positions where one of the strings in Patterns appears as a substring of Text with
    at most d mismatches.
    """
    # to cope with not $-ending text
    if text[-1] != '$':
            text+='$'
            
    s = bwt(text)
    a = suffix_array(text)
    
    def scan_symbol_count(c):
        # update current dict
        current_dict[c] = current_dict.get(c,0)+1
        # copy current dict to the current symbol array
        return current_dict.copy()
    current_dict = {}
    count_symbol = [{}] + [scan_symbol_count(c) for c in s]
    
    # create first_occurence 
    def scan_first_occurence((i,c)):
        if c not in first_occurence:
            first_occurence[c] = i
    first_occurence = {}    
    map(scan_first_occurence,enumerate(sorted(s)))
    
    # move from patterns to seeds
    def pattern_to_seeds(p):
        l = len(p)
        assert l>d
        minsize = l/(d+1)
        cut = range(0,l-minsize+1,minsize)
        cut.append(l)
        seeds = [(p[cut[i-1]:cut[i]],cut[i-1]) for i in range(1,len(cut))]
        return seeds
    
    def seed_positions(seed):
        top = 0
        bottom = len(s) - 1
        while top <= bottom:
            if seed:
                symbol = seed[-1]
                seed = seed[:-1]
                if count_symbol[bottom+1].get(symbol,0) > count_symbol[top].get(symbol,0):   
                        top = first_occurence[symbol] + count_symbol[top].get(symbol,0)
                        bottom = first_occurence[symbol] + count_symbol[bottom+1].get(symbol,0) - 1
                else:
                    return []
            else:
                return [a[i] for i in range(top,bottom+1)]
        return []

    def is_approximately_matching(offset,p):
        mismatches = 0
        for i,c in enumerate(p):
            if (c!=text[offset+i]):
                mismatches += 1
                if mismatches > d:
                    return False
        return True
        
    
    def approximate_pattern_positions(p):
        pattern_positions = set()
        so = pattern_to_seeds(p)
        for (seed,offset) in so:
            candidate_positions = seed_positions(seed)
            for candidate_position in candidate_positions:
                pattern_position = candidate_position - offset
                if pattern_position < 0:
                    # candidate matching before text starts ....
                    continue
                if pattern_position + len(p) > len(text):
                    # candidate matching after text stops ....
                    continue
                if is_approximately_matching(pattern_position,p):
                    # candidate matching with at most d mismatches
                    pattern_positions.add(pattern_position)
        return list(pattern_positions)
       
    pos = []
    for pattern in patterns:
        pos += approximate_pattern_positions(pattern)
    pos.sort()
    return pos             
    
assert multiple_approximate_pattern_matching('ACATGCTACTTT',['ATT', 'GCC', 'GCTA', 'TATT'],1) == [2, 4, 4, 6, 7, 8, 9]

#fname = 'C:/Users/ngaude/Downloads/dataset_301_7.txt'
#with open(fname, "r") as f:
#    text = f.read().strip().split('\n')
#    s = text[0]
#    p = text[1].split(' ')
#with open(fname+'.out', "w") as f:
#    f.write(' '.join(map(str,better_bw_matching(s,p))))

#fname = 'C:/Users/ngaude/Downloads/dataset_9809_2.txt'
#with open(fname, "r") as f:
#    text = f.read().strip().split('\n')
#    s = text[0]
#    k = int(text[1])
#with open(fname+'.out', "w") as f:
#    f.write('\n'.join(map(lambda (i,j): str(i)+','+str(j),partial_suffix_array(s,k))))

#fname = 'C:/Users/ngaude/Downloads/dataset_303_4.txt'
#with open(fname, "r") as f:
#    text = f.read().strip().split('\n')
#    s = text[0]
#    p = text[1:]
#with open(fname+'.out', "w") as f:
#    f.write(' '.join(map(str,multiple_pattern_matching(s,p))))

fname = 'C:/Users/ngaude/Downloads/dataset_304_6.txt'
with open(fname, "r") as f:
    text = f.read().strip().split('\n')
    s = text[0]
    p = text[1].split(' ')
    d = int(text[2])
    print s,p,d
with open(fname+'.out', "w") as f:
    f.write(' '.join(map(str,multiple_approximate_pattern_matching(s,p,d))))