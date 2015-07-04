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


def peptide_score(peptide,spectral_vector,ref_mass = amino_acid_mass):
    m = [ref_mass[aa] for aa in peptide]
    path = [sum(m[:i]) for i in range(len(m)+1)  ]
    score = sum([spectral_vector[x] for x in path])
    return score

def peptide_identification(spectral_vector,proteome, ref_mass = amino_acid_mass):
    """
    CODE CHALLENGE: Solve the Peptide Identification Problem.
    Given: A space-delimited spectral vector Spectrum' and an amino acid string Proteome.
    Return: A substring of Proteome with maximum score against Spectrum'.
    """
    
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
            score = peptide_score(peptide,spectral_vector,ref_mass)
            if score > best_score:
                best_peptide = peptide
                best_score = score
            # then slide window 
            ia+=1
            ib+=1
            
    return best_peptide


a = '0 0 0 4 -2 -3 -1 -7 6 5 3 2 1 9 3 -8 0 3 1 2 1 8'
proteome = 'XZZXZXXXZXZZXZXXZ'
v = [0,] + map(int,a.split(' '))
assert peptide_identification(v,proteome, toy_mass) == 'ZXZXX'


def psm_search(spectral_vectors,proteome,threshold, ref_mass = amino_acid_mass):
    """
    CODE CHALLENGE: Implement PSMSearch to solve the Peptide Search Problem.
    Given: A set of space-delimited spectral vectors SpectralVectors, an amino acid string
    Proteome, and an integer threshold.
    Return: The set PSMthreshold(Proteome, SpectralVectors).
    """
    psms = set()
    for v in vs:
        peptide = peptide_identification(v,proteome,ref_mass)
        score = peptide_score(peptide,v,ref_mass)
        if score >= threshold:
            psms.add(peptide)
    return psms
    
vts = ['-1 5 -4 5 3 -1 -4 5 -1 0 0 4 -1 0 1 4 4 4','-4 2 -2 -4 4 -5 -1 4 -1 2 5 -3 -1 3 2 -3']
vs = [[0,] + map(int,vt.split(' ')) for vt in vts]
proteome = 'XXXZXZXXZXZXXXZXXZX'
threshold = 5 
s = psm_search(vs,proteome,threshold,toy_mass)
assert s ==  {'XZXZ'}

def spectral_dict_size(v,threshold,T,ref_mass = amino_acid_mass):
    """
    CODE CHALLENGE: Solve the Size of Spectral Dictionary Problem.
    Given: A spectral vector Spectrum', an integer threshold, and an integer max_score.
    Return: The size of the dictionary Dictionarythreshold(Spectrum').
    """
    m = len(v)
    size = np.zeros((m+1,T+1),dtype=int)
    size[0,0]=1
    aam = ref_mass.values()
    for i in range(1,m+1):
        for t in range(0,T+1):
#            scit = [ (i-mj,t-v[i-1]) for mj in aam if (i-mj)>=0 and (t-v[i-1])>=0 and (t-v[i-1])<T+1 ]
#            sit = [ size[i-mj,t-v[i-1]] for mj in aam if (i-mj)>=0 and (t-v[i-1])>=0 and (t-v[i-1])<T+1 ]
#            print i,t,scit
            size[i,t] = sum([size[i-mj,t-v[i-1]] 
                for mj in aam if (i-mj)>=0 and (t-v[i-1])>=0 and (t-v[i-1])<T+1 ])
    res = sum(size[m,t] for t in range(threshold,T+1))
    return res
            
vt = '4 -3 -2 3 3 -4 5 -3 -1 -1 3 4 1 3'
v = map(int,vt.split(' '))
threshold = 1
T = 8
assert spectral_dict_size(v,threshold,T,toy_mass) == 3


def spectral_dict_prob(v,threshold,T,ref_mass = amino_acid_mass):
    """
    CODE CHALLENGE: Solve the Size of Spectral Dictionary Problem.
    Given: A spectral vector Spectrum', an integer threshold, and an integer max_score.
    Return: The size of the dictionary Dictionarythreshold(Spectrum').
    """
    m = len(v)
    size = np.zeros((m+1,T+1))
    size[0,0]=1
    aam = ref_mass.values()
    for i in range(1,m+1):
        for t in range(0,T+1):
#            scit = [ (i-mj,t-v[i-1]) for mj in aam if (i-mj)>=0 and (t-v[i-1])>=0 and (t-v[i-1])<T+1 ]
#            sit = [ size[i-mj,t-v[i-1]] for mj in aam if (i-mj)>=0 and (t-v[i-1])>=0 and (t-v[i-1])<T+1 ]
#            print i,t,scit
            size[i,t] = 1./len(aam)*sum([size[i-mj,t-v[i-1]] 
                for mj in aam if (i-mj)>=0 and (t-v[i-1])>=0 and (t-v[i-1])<T+1 ])
    res = sum(size[m,t] for t in range(threshold,T+1))
    return res
            
vt = '4 -3 -2 3 3 -4 5 -3 -1 -1 3 4 1 3'
v = map(int,vt.split(' '))
threshold = 1
T = 8
assert spectral_dict_prob(v,threshold,T,toy_mass) == 0.375


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

#fname = fpath + 'dataset_11866_2.txt'
#with open(fname, 'r') as f:
#    text = f.read()
#    lines = text.split('\n')
#    v = map(int,lines[0].split(' '))
#    v = [0,] + v
#    proteome = lines[1]
#    peptide = peptide_identification(v,proteome)
#    print peptide

#psms = {'QQCGVHEYFWVSKK','HTNGPDCSQYQLLK','VIAAGAHPADGQGVRGP','NGMPFCCMCWDVVM','AAPVCLQQMQPKAVL','SIAQIMVEYTVHGH','KMARKRHIHKFLSP','NRAEQFDMTKYCV','ADMCRPCQACTGKAFG','CKFADFDSKTMGVITQ','DETTVPHLVCPWHD','IFWVHEMMYHCE','GWKRGTYEIIFCPP','DGQGVRGPHQIILMVR','TCFAAGAHVMRKGCH','DCQNYMLMHMVETG','CYCMFHTNTARGERK'}
#fname = fpath + 'psm_search.txt'
#with open(fname, 'r') as f:
#    text = f.read()
#    lines = text.split('\n')
#    vts = lines[:-2]
#    vs = [[0,] + map(int,vt.split(' ')) for vt in vts]
#    proteome = lines[-2]
#    threshold = int(lines[-1])
#    s = psm_search(vs,proteome,threshold)
#    assert s == psms

#fname = fpath + 'dataset_11866_5.txt'
#with open(fname, 'r') as f:
#    text = f.read()
#    lines = text.split('\n')
#    vts = lines[:-3]
#    vs = [[0,] + map(int,vt.split(' ')) for vt in vts]
#    proteome = lines[-3]
#    threshold = int(lines[-2])
#    s = psm_search(vs,proteome,threshold)
#print s

#vt = '14 -4 -3 -3 5 9 0 14 2 1 -4 6 -1 13 2 -5 13 -8 -8 3 0 -10 14 4 14 14 8 -8 1 3 -10 -2 2 -9 3 6 13 -10 6 -8 12 2 8 -1 -5 -6 -6 10 3 -3 12 -4 14 3 11 14 15 12 -7 -5 -2 11 13 -9 15 -8 -10 5 -8 5 6 -9 -2 7 -6 -1 -2 12 12 -3 9 0 3 0 5 -6 3 3 -7 6 0 -6 5 8 -7 5 3 13 13 -2 -9 0 2 13 13 12 7 -2 -10 -5 -7 7 13 11 14 -4 -9 15 -10 5 -7 -6 -7 -6 11 5 9 8 -4 7 1 -9 12 2 8 12 -6 0 2 -5 10 11 14 15 -1 3 -3 3 -3 12 15 4 -2 14 13 8 -10 2 -3 0 -6 8 3 10 0 9 10 13 15 6 9 -10 -9 1 -3 -10 8 1 -10 2 -1 14 -3 15 -1 0 1 6 -7 5 12 6 -9 2 1 -2 14 -5 1 -8 -6 11 -5 2 -3 -8 7 -6 -10 8 6 13 -8 -5 -10 12 -5 -9 8 9 0 10 15 -1 4 2 -8 9 1 -9 -6 -8 -1 -1 5 10 -4 7 3 11 4 12 6 6 13 -3 12 -3 1 7 11 6 13 8 3 -6 5 11 4 -1 15 10 -8 -7 0 4 7 5 -4 8 -3 -4 -8 9 -2 -3 13 1 12 4 -1 13 -1 -5 -5 7 7 -7 -5 6 6 -2 -5 7 10 14 11 12 -9 6 -3 4 15 -8 11 -3 -7 5 -4 7 9 15 -9 8 13 6 -2 -3 9 6 5 14 10 -7 -9 -8 10 2 -3 -1 2 3 12 13 6 -2 8 -5 5 -3 -8 10 3 0 12 -7 10 6 15 8 7 -2 8 14 -2 13 -1 8 15 -7 -7 -7 7 -3 -2 5 -4 -3 15 11 -4 9 11 13 15 8 4 -6 7 12 14 6 -10 -5 -9 4 -9 13 -3 0 12 3 12 -5 11 1 15 -8 5 3 -5 7 15 -2 -9 0 0 1 1 -1 -4 -1 5 12 12 -5 8 5 14 12 5 -9 2 -10 -9 4 -2 6 5 -3 -7 7 5 -8 -10 8 -1 7 3 6 -6 14 -8 6 -5 -8 -10 14 -2 12 4 5 -2 9 -4 1 -5 -3 -6 -8 -9 -10 10 4 9 11 -7 6 -4 5 13 -8 -7 -3 10 14 4 10 6 4 0 13 -3 11 -9 2 -8 6 -8 4 -1'
#v = map(int,vt.split(' '))
#threshold = 37
#T = 200
#assert  spectral_dict_size(v,threshold,T) == 330

#vt = '-2 -3 -3 -2 -1 15 10 5 -3 13 -8 3 1 3 -1 -3 4 14 11 5 14 -3 -9 7 -2 -7 -9 13 -5 3 -8 -6 5 -3 7 1 -8 -4 3 11 -6 -8 13 0 6 8 0 -6 5 3 11 1 -7 0 0 9 1 -9 -6 -7 3 -7 2 14 1 1 -3 -4 5 4 7 8 -10 12 -3 6 6 12 5 15 1 13 0 7 9 7 14 15 8 3 13 7 8 -3 -6 -5 1 14 2 3 5 -9 15 8 13 6 5 -3 12 -8 6 3 6 1 -4 12 7 -9 -1 12 1 12 1 5 5 8 -5 -9 10 15 12 15 -7 -9 15 13 0 13 1 5 9 1 -8 15 7 2 8 3 -9 -1 6 2 -10 -8 -2 2 5 6 -4 15 4 -9 -1 15 3 11 -6 2 7 3 14 -1 6 -5 -4 -5 -9 7 -10 13 -4 3 12 4 12 -9 -7 -8 -3 -9 1 -4 10 4 -9 12 -10 1 14 9 7 10 0 13 -10 0 10 8 -4 -1 1 6 10 -10 -7 -8 6 -3 4 -1 -3 -9 11 -1 6 14 6 2 2 11 7 7 -10 -3 -3 10 13 11 8 14 -3 4 -10 12 13 4 -5 4 4 5 -4 -9 -8 -4 -5 12 9 6 11 7 1 -5 -8 1 9 6 -3 9 10 -10 -10 6 -10 7 -5 -8 10 -8 5 -2 -6 -3 8 8 13 11 8 -4 6 7 -2 13 -6 -2 -9 -3 -8 -5 6 0 -3 -4 -7 10 -7 -1 6 6 -10 6 12 3 5 6 -9 11 2 -9 -9 4 2 -6 5 -6 12 -5 -6 6 3 -2 1 -8 10 -9 7 12 -1 -1 6 7 -1 10 3 12 14 10 1 -2 -8 7 14 10 1 -6 7 15 12 -5 12 -9 -6 -1 6 5 5 15 5 4 -10 11 5 8 -8 6 -10 6 4 5 2 5 1 -7 1 14 -2 -8 9 -4 -2 -3 14 -7 9 -2 11 7 13 0 10 10 -4 -3 12 5 3 -4 9 -10 14 2 5 9 0 -3 -10 -4 -3 -4 4 -2 10 9 -1 14 4 6 -2 13 -6 12 11 -10 13 15 11 5 15 8 8 0 -1 10 -3 3 -8 4 -10 4 13 0 4 -5 0 6 0 8 -7 2 1 15 13 4 8 5 -5 -4 -7 -2 13 14 3 -6 -5 8 -4 9 1 -2 12 -6 -10 7 -8 6 9 -1 1 3 15 3 -7 2 9 13 -7 14 11 13 2 10'
#v = map(int,vt.split(' '))
#threshold = 33
#T = 200
#print 'res=',spectral_dict_size(v,threshold,T)

#vt = '-10 11 3 10 11 12 -6 -5 4 4 -2 9 6 -8 9 -6 -1 10 -6 14 4 13 1 -6 5 -7 13 0 -1 12 -2 11 7 -10 9 13 14 -7 7 -9 -6 4 14 2 -9 1 12 13 15 6 15 13 -6 -10 -10 -8 -8 -7 -10 -7 -6 -4 6 9 -6 7 11 -1 -8 1 9 -5 6 7 -3 -10 -9 -1 4 7 7 -6 14 -6 12 15 7 8 11 -5 8 -8 12 -3 -1 -7 -6 9 13 12 -3 7 7 6 3 1 2 4 10 11 -10 -3 14 9 6 8 -9 1 5 -6 -8 5 -7 6 -6 -7 4 1 -3 7 5 10 11 12 0 -10 12 13 11 3 9 8 -10 9 -8 0 15 4 1 1 -4 12 2 4 0 15 -10 4 -10 -10 6 -5 -5 0 10 -5 8 1 14 6 -3 12 9 -7 -4 -9 -9 7 2 6 4 -10 -9 8 -4 -5 0 7 -4 -3 5 12 -10 3 -6 -10 6 10 -6 3 -5 15 4 14 -1 10 -9 13 11 -7 -5 -3 14 15 6 -3 -8 -5 0 12 0 12 2 8 -1 6 2 4 -6 3 11 -4 -10 1 -5 0 14 -5 -6 -1 15 13 12 -10 6 4 0 14 -1 5 15 13 4 -6 13 12 7 14 6 15 10 -9 1 -8 10 9 6 6 2 9 -2 5 11 -4 -6 -10 -7 10 9 8 -6 1 -8 2 -1 -1 -4 -2 0 9 11 -6 9 11 5 5 14 7 -10 14 -4 7 4 14 14 14 8 2 5 14 -4 13 7 10 14 -7 -6 11 -7 -2 -6 -3 1 -7 7 10 15 -6 -2 0 14 1 9 -7 5 -3 -5 5 -5 0 -4 1 3 11 9 -4 -3 -4 0 1 -4 15 -8 -3 0 0 11 -9 11 5 -9 1 -1 -7 -3 8 -9 11 5 4 4 -7 11 -1 -4 -5 7 -7 7 3 6 13 -1 11 -3 13 11 4 3 2 3 0 12 -6 3 12 -10 -8 -9 12 -2 12 5 -3 5 11 5 1 -2 3 5 1 11 6 -6 -2 0 -7 15 14 15 -10 0 6 13 9 10 -2 10 2 8 6 -6 5 -2 1 13 8 14 1 -4 11 11 -8 0 8 5 5 9 -1 -7 3 15 -7 -8 -3 11 9 0 10 2 1 13 4 0 -6 15 15 -1 10 3 1 2'
#v = map(int,vt.split(' '))
#threshold = 30
#T = 200
#assert round(spectral_dict_prob(v,threshold,T),6) == round(0.00132187890625,6)

vt = '-5 -4 2 11 9 -3 4 1 11 -10 -8 -5 7 -5 14 -5 -2 6 11 -3 12 -6 10 13 3 10 -7 -4 -6 2 7 1 -6 5 15 12 14 -7 11 -7 6 12 9 -5 -10 6 -4 -9 -2 8 4 11 -4 -4 6 3 2 6 -8 2 -7 3 9 7 15 -1 4 6 -1 -9 4 -6 -10 -3 6 8 4 -4 -9 -9 11 -10 5 -10 0 5 9 15 -7 -8 -1 -10 1 3 7 -5 -3 1 -10 1 -6 3 -9 -5 -8 -1 9 -1 7 10 3 -9 -10 14 -9 12 -3 -2 10 -3 -1 -7 10 14 10 -10 13 4 -5 -2 7 9 1 10 -5 13 -6 8 15 10 11 11 -4 -4 6 1 14 -2 -10 -4 3 15 14 1 3 -4 7 -2 9 -10 3 14 1 -2 -3 -9 13 -4 15 0 8 -8 0 15 7 13 -4 10 -5 11 4 9 10 1 6 -4 9 -4 9 -2 3 5 14 3 -8 15 -7 13 -2 0 -4 10 8 12 6 5 -9 1 12 -1 0 3 5 -8 -9 5 4 13 11 13 -9 5 7 4 -4 5 -10 10 15 0 2 9 -8 10 14 -7 7 14 12 -3 0 -2 -4 -10 3 11 0 0 13 15 -5 10 13 4 -8 7 -5 -4 -9 6 -10 15 -10 12 -8 1 -9 3 2 13 9 15 12 9 -2 14 9 14 -1 -9 -5 8 14 -2 -10 14 7 -9 5 15 -7 8 0 8 -6 6 -7 -8 8 0 -5 14 12 13 14 13 -8 -3 4 8 6 6 -9 -8 15 -3 0 9 7 6 1 -1 -8 14 7 3 1 14 14 -1 13 6 1 0 2 2 1 14 -7 -5 15 6 -4 2 14 13 15 8 13 -8 -5 -5 -5 15 11 -10 -8 -2 -3 5 -3 6 6 -5 9 12 5 -6 5 2 12 -7 4 15 7 13 -9 10 13 -5 13 -3 15 -6 1 4 5 14 0 15 -4 -7 -4 -6 3 13 3 12 3 6 15 -7 -8 11 -4 0 7 6 3 -5 -10 -2 8 -7 13 -6 -4 -5 -10 0 2 12 8 2 8 -7 1 4 -5 11 15 15 2 -6 0 -1 2 -6 11 11 -3 -8 1 -2 -9 2 10 -5 -5 -7 2 12 7 7 11 9 -8 -8 11 -6 2 8 4 4 -5 1 -1 7 7 -1 -3 1 12 7 -1 12'
v = map(int,vt.split(' '))
threshold = 34
T = 200
print spectral_dict_prob(v,threshold,T)