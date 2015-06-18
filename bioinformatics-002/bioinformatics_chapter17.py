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

def spectrum_graph(spectrum, ref_mass = amino_acid_mass):
    """
    CODE CHALLENGE: Construct the graph of a spectrum.
    Given: A space-delimited list of integers Spectrum.
    Return: Graph(Spectrum).
    """
    spectrum.append(0)
    spectrum.sort()
    raa = {v:k for k,v in ref_mass.iteritems()}
    aav = ref_mass.values()
    aamax = max(aav)
    adj = []
    for i,si in enumerate(spectrum):
        for j,sj in enumerate(spectrum[i+1:]):
            if sj-si > aamax:
                break
            if (sj-si) in raa:
                adj.append((si,sj,raa[sj-si]))
    return adj

def ideal_spectrum(peptide, ref_mass = amino_acid_mass):
    """
    return an ideal spectrum from peptide
    """
    li = [ ref_mass[aa] for aa in peptide]
    n = len(li)
    spectrum = []
    for i in range(0,n+1):
            spectrum.append(sum(li[0:i]))
            spectrum.append(sum(li[i:]))
    return spectrum


def spectrum_decoding(spectrum, ref_mass = amino_acid_mass):
    """
    CODE CHALLENGE: Solve the Decoding an Ideal Spectrum Problem.
    Given: A space-delimited list of integers Spectrum.
    Return: An amino acid string that explains Spectrum.
    """
    adj = spectrum_graph(spectrum, ref_mass)
    g = {}
    for a,b,c in adj:
        g.setdefault(a,[]).append((b,c))
    source = 0
    sink = max(spectrum)
    paths = [[(source,''),],]
    while len(paths)>0:
        npaths= []
        for p in paths:
            e = p[-1]
            if e[0] in g:
                for ne in g[e[0]]:
                    np = p[:]
                    np.append(ne)
                    if ne[0] == sink:
                        # check if solution found ....
                        peptide = [x[1] for x in np][1:]
                        ispectrum = ideal_spectrum(peptide,ref_mass)
                        ispectrum.sort()
                        if set(ispectrum) == set(spectrum):
                            return peptide
                    else:
                        npaths.append(np)
        paths = npaths
    return None
    
text = '57 71 154 185 301 332 415 429 486'
spectrum = map(int,text.split(' '))
peptide = spectrum_decoding(spectrum)
assert ''.join(peptide) == 'GPFNA'

text = '103 131 259 287 387 390 489 490 577 636 690 693 761 840 892 941 1020 1070 1176 1198 1247 1295 1334 1462 1481 1580 1599 1743 1762 1842 1861 2005 2024 2123 2142 2270 2309 2357 2406 2428 2534 2584 2663 2712 2764 2843 2911 2914 2968 3027 3114 3115 3214 3217 3317 3345 3473 3501 3604'
spectrum = map(int,text.split(' '))
peptide = spectrum_decoding(spectrum)
assert ''.join(peptide) == 'CRQCSLAMQRASQHYVYVWPQETFGFVCRM'

def peptide_vector(peptide, ref_mass = amino_acid_mass):
    """
    CODE CHALLENGE: Solve the Converting a Peptide into a Peptide Vector Problem.
    Given: An amino acid string P.
    Return: The peptide vector of P (in the form of space-separated integers).
    """
    li = [ ref_mass[aa] for aa in peptide]
    n = len(li)
    pm = [sum(li[:i]) for i in range(1,n+1)]
    P = np.zeros(max(pm),dtype = int)
    for e in pm:
        P[e-1]=1
    return P

   
peptide = 'XZZXX'
P = peptide_vector(peptide,toy_mass)
assert ' '.join(map(str,P)) == '0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 0 1'

def vector_peptide(P, ref_mass = amino_acid_mass):
    """
    CODE CHALLENGE: Solve the Converting a Peptide Vector into a Peptide Problem.
    Given: A space-delimited binary vector P
    Return: An amino acid string whose binary peptide vector matches P. For masses
    with more than one amino acid, any choice may be used.
    """
    pm = [i+1 for i,v in enumerate(P) if v==1]
    pm_max = max(pm)
    sm = [pm_max - e for e in pm[:-1]]
    peptide = spectrum_decoding(pm+sm,ref_mass)
    return peptide[::-1]
    
text = '0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 0 1'
P = map(int,text.split(' '))
assert ''.join(vector_peptide(P,toy_mass)) == 'XZZXX'


def peptide_sequencing(spectral_vector, ref_mass = amino_acid_mass):
    """
    CODE CHALLENGE: Solve the Peptide Sequencing Problem.
    Given: A space-delimited spectral vector Spectrum'.
    Return: An amino acid string with maximum score against Spectrum'. 
    For masses with more than one amino acid, any choice may be used.
    """
    raa = {v:k for k,v in ref_mass.iteritems()}
    n = len(spectral_vector)
    spectrum = range(1,n)
    adj = spectrum_graph(spectrum, ref_mass)
    g = {}
    for a,b,c in adj:
        g.setdefault(a,[]).append((b,c))
    source = 0
    sink = max(spectrum)
    best_scored_paths = [(None,None),]*n
    best_modified = True
    best_scored_paths[0]=(0,[source,])
            
    def path_score(path):
        score = sum([spectral_vector[x] for x in path])
        return score
        
#    sol = 'GGPGGPGGAGG'
#    solm = [ref_mass[e] for e in sol]
#    path = [sum(solm[:i]) for i in range(1,len(solm)+1)]
#    print 'path',path
#    print 'sink',sink
#    print 'sol score',path_score(path)
        
    iteration = 0
    while best_modified == True:
        print 'iteration',iteration
        iteration += 1
        best_modified = False
        nbest_scored_paths = best_scored_paths[:]
        for i,(s,p) in enumerate(best_scored_paths):
            if not p:
                # nothing to iterate on, let's continue
                continue
            e = p[-1]
            if e == sink:
                continue
            if e in g:
                for ne,aa in g[e]:
                    np = p[:]
                    np.append(ne)
                    score = path_score(np)
                    nscore = nbest_scored_paths[ne][0]
                    if score > nscore:
                        nbest_scored_paths[ne] = (score,np)
                        best_modified = True
#                        print 'best[',ne,']=',score,' with ',np
        best_scored_paths = nbest_scored_paths
#        print '------------------------------------------------------------'
#        for i,(s,p) in enumerate(best_scored_paths):
#                if s>=0:
#                    print i,'=',s
#        print '------------------------------------------------------------'

#    print '------------------------------------------------------------'
#    for i,(s,p) in enumerate(best_scored_paths):
#        if s>=0:
#            print i,'=',s
#    print '------------------------------------------------------------'        

    best_score,best_path = best_scored_paths[sink] 
    best_peptide = ''
    print 'best_path',best_path,best_score
    print '---\n'
    for i in range(1,len(best_path)):
        aam = best_path[i]-best_path[i-1]
        assert aam in raa
        best_peptide += raa[aam]
    return best_peptide

text = '0 0 0 4 -2 -3 -1 -7 6 5 3 2 1 9 3 -8 0 3 1 2 1 8'
v = [0,] + map(int,text.split(' '))
assert ''.join(peptide_sequencing(v, toy_mass)) == 'XZZXX'


text = '29 20 2 -14 -4 -10 -4 5 16 20 -12 -1 -9 11 -11 12 3 -1 0 3 21 3 -2 10 11 -11 15 17 2 4 8 -19 28 28 29 1 21 27 -15 25 -15 10 10 26 -9 -13 7 -6 9 27 -3 2 -12 0 20 26 -14 15 29 10 30 -17 25 21 -6 26 25 24 6 9 29 -13 27 16 -5 27 25 -3 -20 3 12 27 5 29 3 -13 9 6 12 14 -14 -17 -8 -13 17 6 20 0 -20 -7 -4 -12 18 7 11 3 -8 23 0 -6 27 6 -20 6 1 15 -14 -20 -3 22 -13 6 10 10 -18 -6 -1 -14 8 16 -4 -12 -12 -11 30 11 -4 -11 10 -15 23 -19 27 -17 -3 -3 13 12 -19 5 22 -8 19 21 -6 21 -20 -6 4 -20 -5 17 0 2 13 24 22 6 20 -3 17 -12 24 -18 5 -12 -9 -17 17 -3 29 9 -12 1 -17 -4 8 2 17 -20 28 -15 -11 26 16 19 4 23 -18 22 -15 1 16 14 30 3 12 -5 27 14 -5 29 5 13 16 -3 4 -9 -20 11 11 12 13 2 13 -2 -15 3 4 -20 28 -8 4 28 5 28 16 -3 3 19 17 6 -5 6 -15 6 15 -8 29 22 -3 -5 -5 16 24 14 26 -19 13 -17 22 16 -20 -7 14 -11 -10 -19 30 14 28 20 14 12 -11 -19 -4 -19 -20 22 2 -5 9 -10 -4 -12 15 8 22 27 5 9 18 -13 27 27 -12 8 28 29 -12 -9 29 -15 -7 7 18 -6 -4 -15 -11 23 25 27 -8 3 22 29 7 16 -8 11 -17 -9 5 22 4 -17 7 -18 2 0 -8 2 0 3 -6 23 -3 -17 21 9 -6 7 3 20 9 24 -11 21 1 25 30 -17 -11 12 -13 8 7 -12 -17 0 -1 19 20 -11 13 -6 -14 6 -4 27 6 -18 -4 21 20 -11 21 16 -15 2 -8 8 24 24 -15 -12 -5 -20 -15 27 29 25 -19 -20 14 -16 7 18 -15 25 11 -12 -5 1 -4 12 14 -3 -11 14 -14 25 4 21 -2 -1 -17 23 -3 -9 19 26 8 -18 2 3 20 19 -15 -20 3 26 -3 11 -14 8 25 -16 8 8 2 23 29 -10 28 -16 13 -1 11 22 -20 8 18 2 -16 -17 18 30 0 -19 23 -6 -10 11 24 16 27 19 -16 -17 -9 -8 -5 1 5 0 15 9 24 10 18 10 -7 -9 12 10 21 27 -10 4 -13 -17 -1 3 -17 0 -7 -20 20 29 -13 -11 -4 -12 11 3 25 26 20 27 14 29 17 19 -2 8 17 -8 -8 14 -15 -5 0 23 0 23 22 24 13 1 25 -17 -16 7 18 4 24 -9 10 -12 17 8 28 3 29 -2 26 -8 -3 22 23 -19 1 18 -8 -4 21 29 29 6 -5 5 24 -9 -12 -9 -10 3 5 0 -7 23 22 13 11 16 -20 28 -8 -17 -3 13 13 24 8 13 14 6 -5 -3 20 24 -19 14 -9 -6 16 3 28 -20 -18 11 -18 15 7 29 5 11 -14 21 6 16 -4 16 -10 -17 7 10 2 25 -16 1 16 -19 20 5 1 -3 5 11 21 21 -1 6 -11 -7 28 -8 8 -10 3 24 -9 -4 21 -6 1 -1 11 -13 -12 20 20 -13 -14 -20 -19 26 15 -16 -5 -9 10 4 5 -10 23 14 22 9 -16 16 21 9 15 3 30 6 8 22 25 24 22 1 -16 -3 19 6 -4 -7 -6 25 -13 13 15 21 10 30 -12 19 -1 -2 -19 14 29 -16 12 -17 -8 -12 5 8 25 18 22 7 14 13 25 -20'
v = map(int,text.split(' '))
peptide = peptide_sequencing(v)
assert ''.join(peptide) == 'GGPGGPGGAGG'

############################################################
fpath = 'C:/Users/ngaude/Downloads/'
#fpath = '/home/ngaude/Downloads/'
#fpath = 'C:/Users/Utilisateur/Downloads/'
############################################################



#fname = fpath + 'dataset_11813_2.txt'
#with open(fname, 'r') as f:
#    text = f.read()
#    lines = text.split('\n')
#    spectrum = map(int,lines[0].split(' '))
#    adj = spectrum_graph(spectrum)
#with open(fname+'.out','w') as f:
#    for a,b,c in adj:
#        f.write(str(a)+'->'+str(b)+':'+c+'\n')

#fname = fpath + 'dataset_11813_4.txt'
#with open(fname, 'r') as f:
#    text = f.read()
#    lines = text.split('\n')
#    spectrum = map(int,lines[0].split(' '))
#    peptide = spectrum_decoding(spectrum)
#    print ''.join(peptide)

#fname = fpath + 'dataset_11813_6.txt'
#with open(fname, 'r') as f:
#    text = f.read()
#    lines = text.split('\n')
#    peptide = lines[0]
#    P = peptide_vector(peptide)
#with open(fname+'.out','w') as f:
#    f.write(' '.join(map(str,P)))


#fname = fpath + 'dataset_11813_8.txt'
##fname = fpath + 'peptide_vector_to_peptide.txt'
#with open(fname, 'r') as f:
#    text = f.read()
#    lines = text.split('\n')
#    P = map(int,lines[0].split(' '))
#    peptide = vector_peptide(P)
#    print ''.join(peptide)

fname = fpath + 'dataset_11813_10.txt'
with open(fname, 'r') as f:
    text = f.read()
    lines = text.split('\n')
    v = map(int,lines[0].split(' '))
    v = [0,] + v
    peptide = peptide_sequencing(v)
    print 'peptide_sequencing='+''.join(peptide)