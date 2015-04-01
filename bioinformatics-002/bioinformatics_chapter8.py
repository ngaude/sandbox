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
    return sorted(range(len(s)), key=lambda i: s[i:])

assert suffix_array('AACGATAGCGGTAGA$') == [15, 14, 0, 1, 12, 6, 4, 2, 8, 13, 3, 7, 9, 10, 11, 5]


########################
#
#fname = 'C:/Users/ngaude/Downloads/dataset_310_2.txt'
#with open(fname, "r") as f:
#    text = f.read().strip()
#s = ', '.join(map(str,suffix_array(text)))
#with open(fname+'.out', "w") as f:
#    f.write(s)