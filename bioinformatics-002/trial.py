# -*- coding: utf-8 -*-
"""
Created on Sun Nov 09 16:43:13 2014

@author: ngaude
"""


amino_acid_mass = {'G' : 57, 'A' : 71, 'S' : 87, 'P' : 97, 'V' : 99, 'T' : 101, 'C' : 103,
    'I' : 113, 'L' : 113, 'N' : 114, 'D' : 115, 'K' : 128, 'Q' : 128, 'E' : 129,'M' : 131,
    'H' : 137, 'F' : 147, 'R' : 156, 'Y' : 163, 'W' : 186}

def peptide_masses(peptide):
    '''
    convert peptite string to a list of masses
    '''
    global amino_acid_mass
    return map(lambda k:amino_acid_mass[k],list(peptide))

def spectrum_list(s):
    ''' 
    convert a spectrum string into a list of int
    '''
    return map(int,s.split(' '))
    

def score(pep, spec):
    s = spectrum(subpeptides_n, pep)
    sp = list(spec)
    output = 0
    for x in s:
        if x in sp:
            output += 1
            sp.remove(x)
    return output

def linear_score(pep, spec):
    s = spectrum(linear_subpeptides, pep)
    sp = list(spec)
    output = 0
    for x in s:
        if x in sp:
            output += 1
            sp.remove(x)
    return output

def subpeptides_n(pep):
    p = list(pep)
    n = len(p)
    output = [pep]
    p_extended = p + p[:n-2]
    for i in range(n):
        for j in range(n-1):
            output += [p_extended[i:i+j+1]]
    return output 

def linear_subpeptides(pep):
    output = []
    n = len(pep)
    for i in range(n):
        for j in range(n-i):
            output += [pep[i:i+j+1]]
    return output

def spectrum(function_spectrum, pep):
    subs = function_spectrum(pep)
    return list(sorted([sum(x) for x in subs] + [0]))
    
s = '0 97 97 129 129 194 203 226 226 258 323 323 323 355 403 452'
ss = spectrum_list(s)
p = 'PEEP'
pp = peptide_masses(p)
print 'linear_score',linear_score(pp,ss)
print '----------'
print p,pp,spectrum(linear_subpeptides, pp)
print s,ss
print '----------'
print pp,'subpeptide=',subpeptides_n(pp)