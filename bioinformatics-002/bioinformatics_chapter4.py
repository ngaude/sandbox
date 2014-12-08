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
    
assert composition(5, 'CAATCCAAC') == ['AATCC', 'ATCCA', 'CAATC', 'CCAAC', 'TCCAA']


#fname = 'C:/Users/ngaude/Downloads/dataset_197_3.txt'
#(k, text)  = (l[:-1] for l in open('C:/Users/ngaude/Downloads/dataset_197_3.txt'))
#o = '\n'.join(composition(int(k),text))
#with open(fname+'.out', "w") as f: f.write(o)