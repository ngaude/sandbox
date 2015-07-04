# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 20:11:54 2015

@author: ngaude
"""

import numpy as np

amino_acid_mass = {'G' : 57, 'A' : 71, 'S' : 87, 'P' : 97, 'V' : 99, 'T' : 101, 'C' : 103,
    'I' : 113, 'L' : 113, 'N' : 114, 'D' : 115, 'K' : 128, 'Q' : 128, 'E' : 129,'M' : 131,
    'H' : 137, 'F' : 147, 'R' : 156, 'Y' : 163, 'W' : 186}
    
toy_mass = {'X':4, 'Z':5}

#def spectrum_graph(spectrum, ref_mass = amino_acid_mass):
#    """
#    CODE CHALLENGE: Construct the graph of a spectrum.
#    Given: A space-delimited list of integers Spectrum.
#    Return: Graph(Spectrum).
#    """
#    spectrum.append(0)
#    spectrum.sort()
#    raa = {v:k for k,v in ref_mass.iteritems()}
#    aav = ref_mass.values()
#    aamax = max(aav)
#    adj = []
#    for i,si in enumerate(spectrum):
#        for j,sj in enumerate(spectrum[i+1:]):
#            if sj-si > aamax:
#                break
#            if (sj-si) in raa:
#                adj.append((si,sj,raa[sj-si]))
#    return adj



def peptide_identification(spectral_vector,proteome, ref_mass = amino_acid_mass):
    """
    CODE CHALLENGE: Solve the Peptide Identification Problem.
    Given: A space-delimited spectral vector Spectrum' and an amino acid string Proteome.
    Return: A substring of Proteome with maximum score against Spectrum'.
    """
    def peptide_score(peptide):
        m = [ref_mass[aa] for aa in peptide]
        path = [sum(m[:i]) for i in range(len(m)+1)  ]
        score = sum([spectral_vector[x] for x in path])
        return score
    
    best_peptide = ''
    best_score = None
    # convert proteome into list of sum of its amino acid masses
    plm = [ref_mass[aa] for aa in proteome]
    plsm = [sum(plm[:i]) for i in range(len(plm)+1)  ]
    
    # search for proteome substring matching len of spectral vector
    lsv = len(spectral_vector)-1
    ia = 0
    ib = 0
    while(ib<len(plsm)):
        ws = plsm[ib]-plsm[ia]
        if (ws > lsv):
            # window size is larger than spectral vector, reduce window
            ia+=1
        elif (ws < lsv):
            # window size is smaller than spectral vector, increase window
            ib+=1
        else:
            # we got a candidate
            # check if it is the beste for now
            peptide = proteome[ia:ib]
#            print ws,lsv,ia,ib,peptide
            score = peptide_score(peptide)
            if score > best_score:
                best_peptide = peptide
                best_score = score
            # then slide window 
            ia+=1
            ib+=1
            
    return best_peptide


text = '0 0 0 4 -2 -3 -1 -7 6 5 3 2 1 9 3 -8 0 3 1 2 1 8'
proteome = 'XZZXZXXXZXZZXZXXZ'
v = [0,] + map(int,text.split(' '))
assert peptide_identification(v,proteome, toy_mass) == 'ZXZXX'




############################################################
fpath = 'C:/Users/ngaude/Downloads/'
#fpath = '/home/ngaude/Downloads/'
#fpath = 'C:/Users/Utilisateur/Downloads/'
############################################################

#fname = fpath + 'peptide_identification.txt'
#with open(fname, 'r') as f:
#    text = f.read()
#    lines = text.split('\n')
#    v = map(int,lines[0].split(' '))
#    v = [0,] + v
#    proteome = lines[1]
#    peptide = peptide_identification(v,proteome)
#    assert peptide == 'KLEAARSCFSTRNE'


fname = fpath + 'dataset_11866_2.txt'
with open(fname, 'r') as f:
    text = f.read()
    lines = text.split('\n')
    v = map(int,lines[0].split(' '))
    v = [0,] + v
    proteome = lines[1]
    peptide = peptide_identification(v,proteome)
    print peptide
