# -*- coding: utf-8 -*-
"""
Created on Sun Nov 02 16:07:51 2014

@author: ngaude
"""

from collections import Counter

import operator as op
def ncr(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom

def pattern_count(text, pattern):
    count = 0
    for i in range(len(text) - len(pattern) + 1):
        if (text[i:i+len(pattern)] == pattern):
            count = count + 1
    return count

def frequent_word(text, k):
    kmer_count = {}
    for i in range(len(text) - k + 1):
        kmer = text[i:i+k]
        kmer_count[kmer] = kmer_count.setdefault(kmer,0)+1
    max_count = max(kmer_count.values())
    most_frequent = []
    for kmer,count in kmer_count.iteritems():
        if (count == max_count):
            most_frequent.append(kmer)
    return most_frequent


def reverse_complement(pattern):
    rev = {'A':'T', 'T':'A', 'G':'C', 'C':'G'}
    reverse = map(lambda c: rev[c], pattern[::-1])
    return ''.join(reverse)


def pattern_matching(pattern, genome):
    ''' 
    Returns all starting positions 
    where pattern appears 
    as a substring of genome
    '''
    match = []
    for i in range(len(genome) - len(pattern) + 1):
        if (genome[i:i+len(pattern)] == pattern):
            match.append(i)
    return match   

def clump_finding(genome,k , L, t):
    '''    
    returns all distinct k-mers 
    forming (L, t)-clumps in Genome :
    e.g there is a substring of Genome of length L 
    where this k-mer appears at least t times
    '''
    clump_kmer = []
    kmer_count = {}
    for i in range(len(genome) - L + 1):
#        if (i % (len(genome)/100+1) == 0): 
#            percent =  int((float(i)/len(genome))*100)
#            print percent        
        if (i==0):
            # init kmer-hashtable on first round
            for j in range(L - k + 1):
                kmer = genome[j:j+k]
                kmer_count[kmer] = kmer_count.setdefault(kmer,0)+1
        else:
            # update kmer-hashtable on next round
            o_kmer = genome[i-1:i+k-1]
            i_kmer = genome[i+L-k:i+L]
            if (o_kmer in kmer_count): 
                kmer_count[o_kmer] -= 1
            kmer_count[i_kmer] = kmer_count.setdefault(i_kmer,0)+1
        # remove empty slots from kmer_hashtable
        kmer_count = {k: v for k, v in kmer_count.iteritems() if v>0}
        
        for kmer,count in kmer_count.iteritems():
            if (count>=t):
                clump_kmer.append(kmer)
#        print "-------------"
#        print genome[i:i+L]
#        print kmer_count
    return set(clump_kmer)

def skew(genome):
    '''
    returns the difference of
    count of Guanine nucleotides minus 
    count of Cytosine nucleotides.
    '''
    sk = []
    d = {'C':-1, 'G':1}
    count = 0
    sk.append(0)
    for n in genome:
        count = count + d.setdefault(n,0)
        sk.append(count)
    return sk
     
def min_skew(genome):
    '''
    given a genome, returns all positions 
    where skew minimum is attained
    '''
    sk = skew(genome)
    min_sk = min(sk)
    msk = [i for i in range(len(sk)) if (sk[i]==min_sk)]
    return msk

def hamming_distance(a, b):
    '''
    compute the Hamming distance between two strings.
    '''
    assert len(a) == len(b)
    count = len(a)
    for i in range(len(a)):
        count -= a[i]==b[i]
    return count
    
def approx_pattern_matching(pattern, genome, d):
    ''' 
    Returns all starting positions 
    where pattern appears 
    as an approximate substring of genome
    with a at most 'd' character difference
    '''
    match = []
    for i in range(len(genome) - len(pattern) + 1):
        if (hamming_distance(genome[i:i+len(pattern)],pattern)<=d):
            match.append(i)
    return match   
    
def approx_pattern_count(text, pattern, d):
    res = approx_pattern_matching(pattern, text, d)
    return len(res)

def approx_frequent_word_slow(text, k, d):
    '''
    Find the most frequent k-mers 
    with at most d mismatches 
    in text string
    '''
    kmer_count = {}
    # iterate over all d-mer : from 0 to 4^k-1
    def itokmer(number,k):
        a = ['A', 'C', 'G', 'T']
        kmer = [None] * k
        for i in range(k):
            kmer[k-i-1] = a[number&3]
            number = number >> 2
        return ''.join(kmer)
    max_count = 1
    for i in range(1 << 2*k):
#        if (i % 10000 == 0): print float(i)/(1<<2*k)        
        kmer = itokmer(i, k)
        count = approx_pattern_count(text, kmer, d)
        if (count>=max_count):
            kmer_count[kmer] = count
    max_count = max(kmer_count.values())
    most_frequent = []
    for kmer,count in kmer_count.iteritems():
        if (count == max_count):
            most_frequent.append(kmer)
    return most_frequent

def approx_frequent_word(text, k, d):
    '''
    Find the most frequent k-mers 
    with at most d mismatches 
    in text string
    '''
    def mutation(prefix,suffix,m):
        ''' 
        return list of suffix-string 
        with at least m-mutation
        prefixed with prefix-string
        '''
        if (len(suffix)==0):
            return [prefix]
        
        res = []
        if (m>0):
            if (suffix[0] != 'A'):
                res += mutation(prefix+'A',suffix[1:],m-1)
            if (suffix[0] != 'C'):
                res += mutation(prefix+'C',suffix[1:],m-1)
            if (suffix[0] != 'G'):
                res += mutation(prefix+'G',suffix[1:],m-1)
            if (suffix[0] != 'T'):
                res += mutation(prefix+'T',suffix[1:],m-1)
        res += mutation(prefix+suffix[0],suffix[1:],m)
        return res
    # compute  exact kmer count from original text
    exact_kmer_count = {}
    for i in range(len(text) - k + 1):
        kmer = text[i:i+k]
        exact_kmer_count[kmer] = exact_kmer_count.setdefault(kmer,0)+1
    # extrapolate to approximate kmer count 
    # using kmer all possible mutation 
    approx_kmer_count = {}
    for exact_kmer,count in exact_kmer_count.iteritems():
        kmer_mutations = mutation('',exact_kmer,d)
        for kmer_mutant in kmer_mutations:
            approx_kmer_count[kmer_mutant] = approx_kmer_count.setdefault(kmer_mutant,0)+count
    # return the top approx kmer 
    max_count = max(approx_kmer_count.values())
    most_frequent = []
    for kmer,count in approx_kmer_count.iteritems():
        if (count == max_count):
            most_frequent.append(kmer)
    return most_frequent

def approx_frequent_word_or_reverse(text, k, d):
    '''
    Find the most frequent k-mers 
    with at most d mismatches 
    in text string
    '''
    def mutation(prefix,suffix,m):
        ''' 
        return list of suffix-string 
        with at least m-mutation
        prefixed with prefix-string
        '''
        if (len(suffix)==0):
            return [prefix]
        
        res = []
        if (m>0):
            if (suffix[0] != 'A'):
                res += mutation(prefix+'A',suffix[1:],m-1)
            if (suffix[0] != 'C'):
                res += mutation(prefix+'C',suffix[1:],m-1)
            if (suffix[0] != 'G'):
                res += mutation(prefix+'G',suffix[1:],m-1)
            if (suffix[0] != 'T'):
                res += mutation(prefix+'T',suffix[1:],m-1)
        res += mutation(prefix+suffix[0],suffix[1:],m)
        return res
    # compute  exact kmer count from original text
    exact_kmer_count = {}
    for i in range(len(text) - k + 1):
        kmer = text[i:i+k]
        exact_kmer_count[kmer] = exact_kmer_count.setdefault(kmer,0)+1
    # extrapolate to approximate kmer count 
    # using kmer all possible mutation 
    approx_kmer_count = {}
    for exact_kmer,count in exact_kmer_count.iteritems():
        kmer_mutations = mutation('',exact_kmer,d)
        for kmer_mutant in kmer_mutations:
            approx_kmer_count[kmer_mutant] = approx_kmer_count.setdefault(kmer_mutant,0)+count
            reverse_mutant = reverse_complement(kmer_mutant)            
            approx_kmer_count[reverse_mutant] = approx_kmer_count.setdefault(reverse_mutant,0)+count
    # return the top approx kmer 
    max_count = max(approx_kmer_count.values())
    most_frequent = []
    for kmer,count in approx_kmer_count.iteritems():
        if (count == max_count):
            most_frequent.append(kmer)
    return most_frequent


rna_codon = {'AAA' : 'K', 'AAC' : 'N', 'AAG' : 'K','AAU' : 'N','ACA' : 'T', 'ACC' : 'T', 'ACG' : 'T',
    'ACU' : 'T', 'AGA' : 'R', 'AGC' : 'S', 'AGG' : 'R', 'AGU' : 'S', 'AUA' : 'I', 'AUC' : 'I',
    'AUG' : 'M', 'AUU' : 'I', 'CAA' : 'Q' ,'CAC' : 'H', 'CAG' : 'Q', 'CAU' : 'H', 'CCA' : 'P',
    'CCC' : 'P', 'CCG' : 'P', 'CCU' : 'P', 'CGA' : 'R', 'CGC' : 'R', 'CGG' : 'R', 'CGU' : 'R',
    'CUA' : 'L', 'CUC' : 'L', 'CUG' : 'L', 'CUU' : 'L', 'GAA' : 'E', 'GAC' : 'D', 'GAG' : 'E',
    'GAU' : 'D', 'GCA' : 'A', 'GCC' : 'A', 'GCG' : 'A', 'GCU' : 'A', 'GGA' : 'G', 'GGC' : 'G',
    'GGG' : 'G', 'GGU' : 'G', 'GUA' : 'V', 'GUC' : 'V', 'GUG' : 'V', 'GUU' : 'V', 'UAA' : '-',
    'UAC' : 'Y', 'UAG' : '-', 'UAU' : 'Y', 'UCA' : 'S', 'UCC' : 'S', 'UCG' : 'S', 'UCU' : 'S',
    'UGA' : '-', 'UGC' : 'C', 'UGG' : 'W', 'UGU' : 'C', 'UUA' : 'L', 'UUC' : 'F', 'UUG' : 'L',
    'UUU' : 'F'}

amino_acid_mass = {'G' : 57, 'A' : 71, 'S' : 87, 'P' : 97, 'V' : 99, 'T' : 101, 'C' : 103,
    'I' : 113, 'L' : 113, 'N' : 114, 'D' : 115, 'K' : 128, 'Q' : 128, 'E' : 129,'M' : 131,
    'H' : 137, 'F' : 147, 'R' : 156, 'Y' : 163, 'W' : 186}

def protein_dna_count(s):
    count = 1
    for c in s:
        count *= sum(map(lambda e: e == c, rna_codon.values()))
    return count

def protein_translation(s):
    '''
    translate RNA text into peptide text
    '''
    global rna_codon
    it = (s[i:3+i] for i in range(0, len(s), 3))
    t = ''.join(map(lambda e: rna_codon.setdefault(e,''), it))
    return t
  
def peptide_encoding(dna,peptide):
    '''
    Find substrings of a genome encoding a given amino acid sequence
    '''
    seq = []
    def search_seq(text, reverse = False):
        ttext = protein_translation(text)
        for i in range(len(ttext) - len(peptide) + 1):
            if (ttext[i:i+len(peptide)] == peptide):
                substr = text[i*3:i*3+len(peptide)*3].replace('U', 'T')
                if reverse: substr = reverse_complement(substr)
                seq.append(substr)
    rna = dna.replace('T', 'U')
    rrna = reverse_complement(dna).replace('T', 'U')
    search_seq(rna)
    search_seq(rrna, reverse = True)  
    search_seq(rna[1:])
    search_seq(rrna[1:], reverse = True)        
    search_seq(rna[2:])    
    search_seq(rrna[2:], reverse = True)
    return seq
   
def peptide_masses(peptide):
    '''
    convert peptite string to a list of masses
    '''
    global amino_acid_mass
    return map(lambda k:amino_acid_mass[k],list(peptide))

def peptide_mass_spectrum(pmass, cyclic = True):
    ''' 
    convert list of peptide masses to spectrum
    '''
    s = [0, ]
    ll = list(pmass)    
    n = len(pmass)
    it = None
    if cyclic:
        ll.extend(pmass[:-1])
        s.append(sum(pmass))
        it = [(i,j) for i in range(n) for j in range (i+1,i+n)]
    else:
        it = [(i,j) for i in range(n) for j in range (i+1,n+1)]
        
    for (i,j) in it:
            subpeptide_mass = sum(ll[i:j])
            s.append(subpeptide_mass)
    
    return sorted(s)

def get_spectrum(peptide, cyclic = True):
    '''
    Generate the theoretical spectrum of a cyclic peptide.
    '''
    return peptide_mass_spectrum(peptide_masses(peptide), cyclic)


def counting_peptides_with_given_mass(mass):
    '''
    compute the number of peptides of given total mass.
    '''
    aam = sorted(list(set(amino_acid_mass.values())), reverse = True)
    md = {0:1}
    for i in range(min(aam), mass+1):
        for m in aam:
            if i-m in md:
                md[i] = md[i-m] + md.get(i,0)
    return md[mass]

def linear_subpeptide_count(n):
    '''
    compute the number of subpeptides of a given peptide length.
    '''
    return 1 + (n+1)*n/2

def spectrum_consistent(p,s, cyclic = False):
    lsp = peptide_mass_spectrum(p, cyclic = cyclic)
    for i in lsp:
        if not i in s:
            return False
    return True

def cyclopeptide_sequencing(spectrum):
    '''
    find all peptides consistent with a given spectrum
    '''
    lp = [[]]
    res =  []
    lmass = list(set(amino_acid_mass.values()))
    spectrum.sort(reverse = True)
    parent_mass = max(spectrum)
    def expand(a):
        exp = []
        for i in a:
            for j in lmass:
                p = list(i)
                p.append(j)
                exp.append(p)
        return exp
    while lp:
        lp = expand(lp)
        for p in list(lp):
            if sum(p) == parent_mass:
                if spectrum_consistent(p, spectrum, cyclic = True):
                    res.append(p)
                lp.remove(p)
            elif not spectrum_consistent(p, spectrum):
                lp.remove(p)
    return res  
    
#def counting_peptides_with_given_mass(mass):
#    #seq = []
#    def count(capacity, changes, path):
#        if(capacity == 0):
#            #seq.append(path)
#            n = len(path)
#            rr = Counter(path)
#            comb = 1
#            for k,r in rr.iteritems():
#                if k == 113 or k == 128:
#                    comb *= 2^r 
#                comb *= ncr(n, r)
#                n -= r
#            if (len(path)>1): comb /= 2
#            return comb
#        elif (capacity < 0): return 0
#        elif (not changes and capacity>=1 ): return 0
#        else: 
#            l = list(path)
#            r = list(path)
#            r.append(changes[0])
#            return count(capacity, changes[1:], l) + count(capacity - changes[0],changes, r)
#    coins = sorted(amino_acid_mass.values(), reverse = True)
#    return count(mass, coins, [])

def peptide_scoring(peptide, spectrum, cyclic = True):
    '''
    Compute the score of a cyclic peptide against a spectrum.
    '''
    lsp = get_spectrum(peptide, cyclic = cyclic)
    spectrum.sort()
    lsp.sort()
    score = 0
    i = 0
    j = 0
    while i < len(lsp) and j < len(spectrum):
        if (spectrum[j] == lsp[i]):
            j += 1
            i += 1
            score +=1
        elif (spectrum[j] > lsp[i]):
            i += 1
        else:
            j += 1
    return score

def spectral_convolution(spectrum):
    spectrum.sort()
    conv = []
    n = len(spectrum)
    for i in range(n):
        for j in range(i+1,n):
            diff = spectrum[j]-spectrum[i]
            if diff > 0:
                conv.append(diff)
    return conv
            

assert  'CATG' in frequent_word('ACGTTGCATGTCGCATGATGCATGAGAGCT', 4)
assert 'ACCGGGTTTT' ==  reverse_complement('AAAACCCGGT')
assert [1, 3, 9] == pattern_matching('ATAT', 'GATATATGCATATACTT')
assert {'CGACA', 'GAAGA'} == clump_finding('CGGACTCGACAGATGTGAAGAACGACAATGTGAAGACTCGACACGACAGAGTGAAGAGAAGAGGAAACATTGTAA', 5, 50, 4)
assert [11, 24] == min_skew('TAAAGACTGCCGAGAGGCCAACACGAGTGCTAGAACGAGGGGCGTAAACGCGGGTCCGAT')
assert 3 == hamming_distance('GGGCCGTTGGT','GGACCGTTGAC')
assert [6, 7, 26, 27] == approx_pattern_matching('ATTCTGGA','CGCCCGAATCCAGAACGCATTCCCATATTTCGGGACCACTGGCCTCCACGGTACGGACGTCAATCAAAT',3)
assert 4 == approx_pattern_count('TTTAGAGCCTTCAGAGG', 'GAGG', 2)
assert {'GATG', 'ATGC', 'ATGT'} == set(approx_frequent_word('ACGTTGCATGTCGCATGATGCATGAGAGCT', 4, 1))
assert {'ATGT', 'ACAT'} == set(approx_frequent_word_or_reverse('ACGTTGCATGTCGCATGATGCATGAGAGCT', 4, 1))
assert protein_translation('AUGGCCAUGGCGCCCAGAACUGAGAUCAAUAGUACCCGUAUUAACGGGUGA').replace('-','') == 'MAMAPRTEINSTRING'
assert set(peptide_encoding('ATGGCCATGGCCCCCAGAACTGAGATCAATAGTACCCGTATTAACGGGTGA','MA')) == {'ATGGCC', 'GGCCAT', 'ATGGCC'}
assert ' '.join(map(str,get_spectrum('LEQN'))) == '0 113 114 128 129 227 242 242 257 355 356 370 371 484'
assert counting_peptides_with_given_mass(1024) == 14712706211L
assert sorted(cyclopeptide_sequencing(map(int,'0 113 128 186 241 299 314 427'.split(' ')))) == sorted([[186, 128, 113], [186, 113, 128], [128, 186, 113], [128, 113, 186], [113, 186, 128], [113, 128, 186]])
assert peptide_scoring('NQEL',[0, 99, 113, 114, 128, 227, 257, 299, 355, 356, 370, 371, 484]) == 11
assert peptide_scoring('NQEL', get_spectrum('NQEL'), cyclic = False) == 11

#
#s = '0 97 97 129 129 194 203 226 226 258 323 323 323 355 403 452'
#ss = map(int,s.split(' '))
#p = 'MAMA'
#pp = peptide_masses(p)
#print pp,get_spectrum(p)
#print ss

#
#s = '0 57 118 179 236 240 301'
#conv = sorted(spectral_convolution(map(int,s.split(' '))))
#print ' '.join(map(str,conv))
#print Counter(conv)

#s = '0 86 160 234 308 320 382'
#conv = spectral_convolution(map(int,s.split(' ')))
#print ' '.join(map(str,conv))

#
#s = '465 473 998 257 0 385 664 707 147 929 87 450 748 938 998 768 234 722 851 113 700 957 265 284 250 137 317 801 128 820 321 612 956 434 534 621 651 129 421 337 216 699 347 101 464 601 87 563 738 635 386 972 620 851 948 200 156 571 551 522 828 984 514 378 363 484 855 869 835 234 1085 764 230 885'
#conv = spectral_convolution(map(int,s.split(' ')))
#print ' '.join(map(str,conv))
#ok = '87 87 87 87 87 87 87 87 87 87 87 87 87 87 87 87 87 87 87 87 87 87 87 87 87 87 87 87 87 87 87 87 234 234 234 234 234 234 234 234 234 234 234 234 234 234 234 234 234 234 234 234 234 234 234 234 234 234 234 234 113 113 113 113 113 113 113 113 113 113 113 113 113 113 113 113 113 113 113 113 113 113 113 113 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 129 147 147 147 147 147 147 147 147 147 147 147 147 147 147 147 147 147 147 147 147 147 147 250 250 250 250 250 250 250 250 250 250 250 250 250 250 250 250 250 250 250 250 250 250 137 137 137 137 137 137 137 137 137 137 137 137 137 137 137 137 137 137 137 137 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 121 121 121 121 121 121 121 121 121 121 121 121 121 121 121 121 121 121 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 230 230 230 230 230 230 230 230 230 230 230 230 230 230 230 230 230 230 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 101 101 101 101 101 101 101 101 101 101 101 101 101 101 101 101 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 216 216 216 216 216 216 216 216 216 216 216 216 216 216 216 216 257 257 257 257 257 257 257 257 257 257 257 257 257 257 257 257 284 284 284 284 284 284 284 284 284 284 284 284 284 284 284 284 337 337 337 337 337 337 337 337 337 337 337 337 337 337 337 337 378 378 378 378 378 378 378 378 378 378 378 378 378 378 378 378 386 386 386 386 386 386 386 386 386 386 386 386 386 386 386 386 635 635 635 635 635 635 635 635 635 635 635 635 635 635 635 635 16 16 16 16 16 16 16 16 16 16 16 16 16 16 265 265 265 265 265 265 265 265 265 265 265 265 265 265 317 317 317 317 317 317 317 317 317 317 317 317 317 317 347 347 347 347 347 347 347 347 347 347 347 347 347 347 363 363 363 363 363 363 363 363 363 363 363 363 363 363 401 401 401 401 401 401 401 401 401 401 401 401 401 401 464 464 464 464 464 464 464 464 464 464 464 464 464 464 484 484 484 484 484 484 484 484 484 484 484 484 484 484 514 514 514 514 514 514 514 514 514 514 514 514 514 514 534 534 534 534 534 534 534 534 534 534 534 534 534 534 117 117 117 117 117 117 117 117 117 117 117 117 117 34 34 34 34 34 34 34 34 34 34 34 34 49 49 49 49 49 49 49 49 49 49 49 49 136 136 136 136 136 136 136 136 136 136 136 136 249 249 249 249 249 249 249 249 249 249 249 249 321 321 321 321 321 321 321 321 321 321 321 321 336 336 336 336 336 336 336 336 336 336 336 336 385 385 385 385 385 385 385 385 385 385 385 385 421 421 421 421 421 421 421 421 421 421 421 421 435 435 435 435 435 435 435 435 435 435 435 435 465 465 465 465 465 465 465 465 465 465 465 465 473 473 473 473 473 473 473 473 473 473 473 473 601 601 601 601 601 601 601 601 601 601 601 601 651 651 651 651 651 651 651 651 651 651 651 651 41 41 41 41 41 41 41 41 41 41 41 185 185 185 185 185 185 185 185 185 185 185 8 8 8 8 8 8 8 8 8 8 69 69 69 69 69 69 69 69 69 69 79 79 79 79 79 79 79 79 79 79 103 103 103 103 103 103 103 103 103 103 106 106 106 106 106 106 106 106 106 106 144 144 144 144 144 144 144 144 144 144 170 170 170 170 170 170 170 170 170 170 178 178 178 178 178 178 178 178 178 178 208 208 208 208 208 208 208 208 208 208 298 298 298 298 298 298 298 298 298 298 314 314 314 314 314 314 314 314 314 314 434 434 434 434 434 434 434 434 434 434 450 450 450 450 450 450 450 450 450 450 488 488 488 488 488 488 488 488 488 488 522 522 522 522 522 522 522 522 522 522 612 612 612 612 612 612 612 612 612 612 620 620 620 620 620 620 620 620 620 620 621 621 621 621 621 621 621 621 621 621 722 722 722 722 722 722 722 722 722 722 764 764 764 764 764 764 764 764 764 764 571 571 571 571 571 571 571 571 571 14 14 14 14 14 14 14 14 15 15 15 15 15 15 15 15 19 19 19 19 19 19 19 19 20 20 20 20 20 20 20 20 23 23 23 23 23 23 23 23 26 26 26 26 26 26 26 26 27 27 27 27 27 27 27 27 30 30 30 30 30 30 30 30 31 31 31 31 31 31 31 31 60 60 60 60 60 60 60 60 64 64 64 64 64 64 64 64 72 72 72 72 72 72 72 72 78 78 78 78 78 78 78 78 97 97 97 97 97 97 97 97 98 98 98 98 98 98 98 98 133 133 133 133 133 133 133 133 148 148 148 148 148 148 148 148 151 151 151 151 151 151 151 151 152 152 152 152 152 152 152 152 163 163 163 163 163 163 163 163 171 171 171 171 171 171 171 171 187 187 187 187 187 187 187 187 193 193 193 193 193 193 193 193 197 197 197 197 197 197 197 197 235 235 235 235 235 235 235 235 273 273 273 273 273 273 273 273 288 288 288 288 288 288 288 288 306 306 306 306 306 306 306 306 334 334 334 334 334 334 334 334 351 351 351 351 351 351 351 351 355 355 355 355 355 355 355 355 371 371 371 371 371 371 371 371 405 405 405 405 405 405 405 405 483 483 483 483 483 483 483 483 507 507 507 507 507 507 507 507 548 548 548 548 548 548 548 548 551 551 551 551 551 551 551 551 563 563 563 563 563 563 563 563 594 594 594 594 594 594 594 594 714 714 714 714 714 714 714 714 738 738 738 738 738 738 738 738 748 748 748 748 748 748 748 748 768 768 768 768 768 768 768 768 367 367 367 367 367 367 367 367 851 851 851 851 851 851 851 851 57 57 57 57 57 57 57 1 1 1 1 1 1 4 4 4 4 4 4 9 9 9 9 9 9 18 18 18 18 18 18 42 42 42 42 42 42 43 43 43 43 43 43 63 63 63 63 63 63 65 65 65 65 65 65 71 71 71 71 71 71 80 80 80 80 80 80 83 83 83 83 83 83 84 84 84 84 84 84 88 88 88 88 88 88 99 99 99 99 99 99 102 102 102 102 102 102 105 105 105 105 105 105 143 143 143 143 143 143 167 167 167 167 167 167 177 177 177 177 177 177 180 180 180 180 180 180 186 186 186 186 186 186 191 191 191 191 191 191 204 204 204 204 204 204 214 214 214 214 214 214 226 226 226 226 226 226 231 231 231 231 231 231 238 238 238 238 238 238 239 239 239 239 239 239 254 254 254 254 254 254 264 264 264 264 264 264 272 272 272 272 272 272 280 280 280 280 280 280 291 291 291 291 291 291 299 299 299 299 299 299 300 300 300 300 300 300 304 304 304 304 304 304 318 318 318 318 318 318 322 322 322 322 322 322 328 328 328 328 328 328 344 344 344 344 344 344 370 370 370 370 370 370 377 377 377 377 377 377 387 387 387 387 387 387 397 397 397 397 397 397 417 417 417 417 417 417 427 427 427 427 427 427 442 442 442 442 442 442 447 447 447 447 447 447 491 491 491 491 491 491 492 492 492 492 492 492 499 499 499 499 499 499 504 504 504 504 504 504 508 508 508 508 508 508 538 538 538 538 538 538 564 564 564 564 564 564 570 570 570 570 570 570 586 586 586 586 586 586 617 617 617 617 617 617 664 664 664 664 664 664 681 681 681 681 681 681 699 699 699 699 699 699 700 700 700 700 700 700 707 707 707 707 707 707 741 741 741 741 741 741 782 782 782 782 782 782 801 801 801 801 801 801 828 828 828 828 828 828 869 869 869 869 869 869 885 885 885 885 885 885 155 155 155 155 155 243 243 243 243 243 313 313 313 313 313 329 329 329 329 329 443 443 443 443 443 585 585 585 585 585 7 7 7 7 10 10 10 10 12 12 12 12 28 28 28 28 29 29 29 29 35 35 35 35 36 36 36 36 37 37 37 37 38 38 38 38 39 39 39 39 44 44 44 44 46 46 46 46 48 48 48 48 52 52 52 52 53 53 53 53 56 56 56 56 58 58 58 58 67 67 67 67 68 68 68 68 74 74 74 74 86 86 86 86 90 90 90 90 93 93 93 93 94 94 94 94 100 100 100 100 107 107 107 107 110 110 110 110 118 118 118 118 120 120 120 120 149 149 149 149 150 150 150 150 162 162 162 162 164 164 164 164 165 165 165 165 166 166 166 166 169 169 169 169 181 181 181 181 184 184 184 184 188 188 188 188 189 189 189 189 199 199 199 199 201 201 201 201 205 205 205 205 207 207 207 207 215 215 215 215 218 218 218 218 219 219 219 219 220 220 220 220 223 223 223 223 224 224 224 224 227 227 227 227 236 236 236 236 242 242 242 242 246 246 246 246 248 248 248 248 258 258 258 258 260 260 260 260 274 274 274 274 276 276 276 276 277 277 277 277 278 278 278 278 279 279 279 279 286 286 286 286 287 287 287 287 292 292 292 292 294 294 294 294 301 301 301 301 303 303 303 303 308 308 308 308 327 327 327 327 333 333 333 333 335 335 335 335 352 352 352 352 356 356 356 356 360 360 360 360 362 362 362 362 364 364 364 364 61 61 61 61 375 375 375 375 382 382 382 382 383 383 383 383 390 390 390 390 394 394 394 394 404 404 404 404 407 407 407 407 414 414 414 414 415 415 415 415 416 416 416 416 423 423 423 423 430 430 430 430 438 438 438 438 454 454 454 454 457 457 457 457 466 466 466 466 470 470 470 470 472 472 472 472 476 476 476 476 498 498 498 498 500 500 500 500 506 506 506 506 511 511 511 511 518 518 518 518 525 525 525 525 530 530 530 530 533 533 533 533 536 536 536 536 544 544 544 544 552 552 552 552 567 567 567 567 577 577 577 577 578 578 578 578 598 598 598 598 609 609 609 609 613 613 613 613 619 619 619 619 639 639 639 639 655 655 655 655 661 661 661 661 672 672 672 672 673 673 673 673 677 677 677 677 688 688 688 688 691 691 691 691 695 695 695 695 704 704 704 704 723 723 723 723 727 727 727 727 733 733 733 733 750 750 750 750 756 756 756 756 798 798 798 798 820 820 820 820 835 835 835 835 842 842 842 842 855 855 855 855 861 861 861 861 870 870 870 870 897 897 897 897 911 911 911 911 998 998 998 998 217 217 217 391 391 391 451 451 451 517 517 517 625 625 625 11 11 13 13 22 22 24 24 33 33 54 54 55 55 70 70 82 82 91 91 95 95 104 104 109 109 115 115 122 122 126 126 127 127 130 130 131 131 135 135 142 142 159 159 161 161 173 173 174 174 175 175 183 183 190 190 192 192 209 209 210 210 213 213 221 221 222 222 229 229 241 241 256 256 262 262 266 266 267 267 268 268 269 269 275 275 283 283 285 285 293 293 295 295 297 297 305 305 309 309 320 320 326 326 330 330 341 341 343 343 345 345 349 349 353 353 358 358 366 366 372 372 379 379 380 380 393 393 395 395 396 396 399 399 406 406 409 409 412 412 413 413 419 419 420 420 422 422 424 424 426 426 431 431 433 433 445 445 448 448 449 449 456 456 458 458 462 462 469 469 474 474 475 475 477 477 479 479 480 480 481 481 493 493 495 495 503 503 519 519 520 520 523 523 527 527 532 532 535 535 543 543 550 550 553 553 560 560 562 562 566 566 568 568 572 572 575 575 579 579 582 582 587 587 590 590 591 591 592 592 593 593 599 599 604 604 605 605 606 606 608 608 610 610 611 611 627 627 628 628 631 631 636 636 637 637 640 640 645 645 647 647 654 654 663 663 667 667 669 669 679 679 683 683 692 692 698 698 706 706 708 708 713 713 715 715 718 718 719 719 726 726 729 729 732 732 734 734 740 740 742 742 754 754 757 757 772 772 784 784 792 792 800 800 809 809 810 810 816 816 819 819 825 825 837 837 843 843 844 844 847 847 856 856 871 871 929 929 938 938 948 948 956 956 957 957 972 972 984 984 17 1085 139 157 315 359 411 555 653 685 773 791 811 827 829 859 883'
#test = map(int, ok.split(' '))
#test.sort()
#conv.sort()
#print test == conv
#for i in range(0,len(test),10):
#    print conv[i:i+10]
#    print test[i:i+10]
#    print '----------'
#    


#pp = ['CTQ','TCQ','CTV','TCE','QCV','ETC']
#ss = '0 71 99 101 103 128 129 199 200 204 227 230 231 298 303 328 330 332 333'
#s = map(int,ss.split( ))
#print 
#for p in pp:
#    ps = peptide_masses(p)
#    print p, spectrum_consistent(ps,s)
#    print ss
#    print sorted(peptide_mass_spectrum(ps, cyclic = False))
#    print '----------'

#
#s = map(int, '0 57 71 71 71 104 131 202 202 202 256 333 333 403 404'.split( ))
#p = 'MAMA'
#print p,peptide_scoring(p,s)
##
#
#s = map(int, '0 97 97 97 100 129 194 226 226 226 258 323 323 355 393 452'.split( ))
#p = 'PEEP'
#print p,peptide_scoring(p,s, cyclic = False)

#assert peptide_scoring('NQEL',[0, 99, 113, 114, 128, 227, 257, 299, 355, 356, 370, 371, 484], cyclic = False) == 8

#s = '0 71 101 113 131 184 202 214 232 285 303 315 345 416'
#c = 'MTAL'
#print c,' '.join(map(str,get_spectrum(c))) == s
#c = 'TMLA'
#print c,' '.join(map(str,get_spectrum(c))) == s
#c = 'MAIT'
#print c,' '.join(map(str,get_spectrum(c))) == s
#c = 'TLAM'
#print c,' '.join(map(str,get_spectrum(c))) == s
#c = 'IAMT'
#print c,' '.join(map(str,get_spectrum(c))) == s
#c = 'TMIA'
#print c,' '.join(map(str,get_spectrum(c))) == s
#c = 'MLAT'
#print c,' '.join(map(str,get_spectrum(c))) == s
#c = 'TALM'
#print c,' '.join(map(str,get_spectrum(c))) == s

#s = '0 57 57 71 71 71 87 97 97 103 103 113 113 113 114 114 115 115 128 128 128 128 128 129 129 131 137 142 147 147 156 163 163 163 163 163 163 163 170 172 185 185 186 210 216 216 218 225 227 229 234 234 234 234 234 241 242 242 243 250 257 260 261 266 266 270 275 275 276 291 291 298 298 298 300 305 305 310 310 314 317 326 331 333 338 338 339 341 344 345 349 357 357 363 373 376 378 379 379 388 389 394 395 397 397 397 397 401 403 406 412 413 417 420 427 430 438 452 454 460 461 461 466 468 473 473 476 480 482 485 488 496 507 507 509 510 510 514 516 516 517 520 523 525 525 526 532 539 544 545 552 558 559 560 564 567 567 575 579 580 582 583 583 590 597 613 616 623 627 629 631 635 636 637 638 639 639 639 640 645 646 648 654 654 659 677 677 687 687 692 694 695 695 695 695 696 702 707 707 708 711 722 726 730 741 742 745 746 751 751 752 752 752 754 758 768 774 778 779 782 782 790 793 795 805 809 809 822 823 824 824 826 839 840 848 849 850 850 854 855 855 855 857 858 858 866 867 870 874 879 880 882 893 893 893 895 896 911 912 914 921 924 937 937 937 940 941 952 953 953 953 956 961 968 968 970 970 972 979 983 986 989 992 992 995 995 1005 1012 1013 1021 1024 1027 1027 1040 1040 1042 1043 1052 1056 1056 1056 1065 1066 1066 1066 1068 1074 1083 1084 1084 1084 1089 1089 1092 1097 1098 1100 1103 1109 1115 1119 1119 1120 1123 1127 1127 1136 1146 1153 1155 1155 1155 1158 1165 1168 1169 1171 1180 1184 1187 1194 1198 1202 1203 1211 1212 1212 1212 1212 1217 1219 1220 1222 1229 1229 1231 1234 1247 1247 1250 1251 1252 1256 1258 1261 1261 1266 1268 1277 1278 1282 1284 1290 1290 1293 1299 1300 1316 1325 1326 1328 1329 1331 1331 1334 1340 1340 1341 1347 1349 1349 1354 1357 1361 1364 1364 1365 1371 1374 1378 1380 1385 1392 1394 1406 1410 1413 1414 1418 1419 1422 1428 1428 1429 1432 1441 1444 1446 1453 1453 1456 1462 1463 1463 1463 1468 1469 1475 1477 1477 1478 1482 1485 1485 1493 1494 1495 1497 1503 1524 1527 1527 1533 1534 1541 1547 1550 1550 1553 1556 1557 1557 1569 1574 1576 1578 1581 1582 1588 1588 1591 1591 1591 1592 1595 1598 1599 1606 1607 1608 1609 1615 1619 1624 1626 1626 1638 1640 1644 1647 1648 1652 1662 1664 1669 1678 1679 1681 1690 1690 1694 1696 1702 1704 1705 1716 1719 1719 1720 1722 1723 1737 1737 1737 1741 1743 1743 1746 1748 1751 1751 1754 1761 1761 1761 1766 1767 1773 1773 1775 1779 1787 1792 1794 1801 1807 1807 1808 1809 1810 1811 1819 1832 1838 1840 1844 1847 1849 1850 1858 1861 1865 1866 1872 1874 1874 1879 1879 1881 1888 1890 1894 1900 1901 1906 1907 1907 1914 1921 1923 1923 1924 1924 1924 1929 1932 1936 1936 1938 1952 1957 1958 1960 1964 1971 1977 1980 1985 1991 1993 1993 1994 1994 1995 2003 2003 2008 2009 2012 2013 2021 2035 2035 2035 2036 2037 2042 2051 2053 2057 2057 2060 2064 2070 2071 2079 2080 2086 2087 2099 2104 2106 2106 2107 2108 2108 2108 2113 2116 2121 2122 2123 2124 2124 2134 2136 2140 2140 2140 2148 2150 2154 2157 2158 2164 2166 2170 2171 2173 2177 2184 2185 2198 2204 2207 2214 2219 2221 2221 2222 2223 2227 2233 2234 2234 2236 2237 2247 2250 2251 2253 2255 2264 2267 2268 2269 2276 2278 2285 2286 2287 2290 2299 2301 2303 2304 2307 2317 2318 2320 2320 2321 2327 2331 2332 2333 2333 2335 2340 2347 2350 2356 2369 2370 2377 2381 2383 2384 2388 2390 2396 2397 2400 2404 2406 2414 2414 2414 2418 2420 2430 2430 2431 2432 2433 2438 2441 2446 2446 2446 2447 2448 2448 2450 2455 2467 2468 2474 2475 2483 2484 2490 2494 2497 2497 2501 2503 2512 2517 2518 2519 2519 2519 2533 2541 2542 2545 2546 2551 2551 2559 2560 2560 2561 2561 2563 2569 2574 2577 2583 2590 2594 2596 2597 2602 2616 2618 2618 2622 2625 2630 2630 2630 2631 2631 2633 2640 2647 2647 2648 2653 2654 2660 2664 2666 2673 2675 2675 2680 2680 2682 2688 2689 2693 2696 2704 2705 2707 2710 2714 2716 2722 2735 2743 2744 2745 2746 2747 2747 2753 2760 2762 2767 2775 2779 2781 2781 2787 2788 2793 2793 2793 2800 2803 2803 2806 2808 2811 2811 2813 2817 2817 2817 2831 2832 2834 2835 2835 2838 2849 2850 2852 2858 2860 2864 2864 2873 2875 2876 2885 2890 2892 2902 2906 2907 2910 2914 2916 2928 2928 2930 2935 2939 2945 2946 2947 2948 2955 2956 2959 2962 2963 2963 2963 2966 2966 2972 2973 2976 2978 2980 2985 2997 2997 2998 3001 3004 3004 3007 3013 3020 3021 3027 3027 3030 3051 3057 3059 3060 3061 3069 3069 3072 3076 3077 3077 3079 3085 3086 3091 3091 3091 3092 3098 3101 3101 3108 3110 3113 3122 3125 3126 3126 3132 3135 3136 3140 3141 3144 3148 3160 3162 3169 3174 3176 3180 3183 3189 3190 3190 3193 3197 3200 3205 3205 3207 3213 3214 3214 3220 3223 3223 3225 3226 3228 3229 3238 3254 3255 3261 3264 3264 3270 3272 3276 3277 3286 3288 3293 3293 3296 3298 3302 3303 3304 3307 3307 3320 3323 3325 3325 3332 3334 3335 3337 3342 3342 3342 3342 3343 3351 3352 3356 3360 3367 3370 3374 3383 3385 3386 3389 3396 3399 3399 3399 3401 3408 3418 3427 3427 3431 3434 3435 3435 3439 3445 3451 3454 3456 3457 3462 3465 3465 3470 3470 3470 3471 3480 3486 3488 3488 3488 3489 3498 3498 3498 3502 3511 3512 3514 3514 3527 3527 3530 3533 3541 3542 3549 3559 3559 3562 3562 3565 3568 3571 3575 3582 3584 3584 3586 3586 3593 3598 3601 3601 3601 3602 3613 3614 3617 3617 3617 3630 3633 3640 3642 3643 3658 3659 3661 3661 3661 3672 3674 3675 3680 3684 3687 3688 3696 3696 3697 3699 3699 3699 3700 3704 3704 3705 3706 3714 3715 3728 3730 3730 3731 3732 3745 3745 3749 3759 3761 3764 3772 3772 3775 3776 3780 3786 3796 3800 3802 3802 3802 3803 3803 3808 3809 3812 3813 3824 3828 3832 3843 3846 3847 3847 3852 3858 3859 3859 3859 3859 3860 3862 3867 3867 3877 3877 3895 3900 3900 3906 3908 3909 3914 3915 3915 3915 3916 3917 3918 3919 3923 3925 3927 3931 3938 3941 3957 3964 3971 3971 3972 3974 3975 3979 3987 3987 3990 3994 3995 3996 4002 4009 4010 4015 4022 4028 4029 4029 4031 4034 4037 4038 4038 4040 4044 4044 4045 4047 4047 4058 4066 4069 4072 4074 4078 4081 4081 4086 4088 4093 4093 4094 4100 4102 4116 4124 4127 4134 4137 4141 4142 4148 4151 4153 4157 4157 4157 4157 4159 4160 4165 4166 4175 4175 4176 4178 4181 4191 4197 4197 4205 4209 4210 4213 4215 4216 4216 4221 4223 4228 4237 4240 4244 4244 4249 4249 4254 4256 4256 4256 4263 4263 4278 4279 4279 4284 4288 4288 4293 4294 4297 4304 4311 4312 4312 4313 4320 4320 4320 4320 4320 4325 4327 4329 4336 4338 4338 4344 4368 4369 4369 4382 4384 4391 4391 4391 4391 4391 4391 4391 4398 4407 4407 4412 4417 4423 4425 4425 4426 4426 4426 4426 4426 4439 4439 4440 4440 4441 4441 4441 4451 4451 4457 4457 4467 4483 4483 4483 4497 4497 4554'
#p = 'IGQTYQNDQLCKWYFYTAYAARNPQYSEQGLYDHELC'
#a = map(int,s.split(' '))
#print cyclopeptide_scoring(p,a)

#
#i = '0 71 71 97 97 99 113 129 147 147 170 184 194 200 210 218 226 246 281 294 297 299 307 317 323 331 365 378 393 394 396 428 436 446 464 464 478 493 507 507 525 535 543 575 577 578 593 606 640 648 654 664 672 674 677 690 725 745 753 761 771 777 787 801 824 824 842 858 872 874 874 900 900 971'
#lsp = map(int,i.split(' '))
#seq = cyclopeptide_sequencing(lsp)
#print ' '.join(map(lambda k : '-'.join(map(str,k)),seq))

#
#res = map(lambda k : '-'.join(map(str,k)),seq)
#print res
#
#ok = '103-137-71-131-114-113-113-115-99-97 103-97-99-115-113-113-114-131-71-137 113-113-114-131-71-137-103-97-99-115 113-113-115-99-97-103-137-71-131-114 113-114-131-71-137-103-97-99-115-113 113-115-99-97-103-137-71-131-114-113 114-113-113-115-99-97-103-137-71-131 114-131-71-137-103-97-99-115-113-113 115-113-113-114-131-71-137-103-97-99 115-99-97-103-137-71-131-114-113-113 131-114-113-113-115-99-97-103-137-71 131-71-137-103-97-99-115-113-113-114 137-103-97-99-115-113-113-114-131-71 137-71-131-114-113-113-115-99-97-103 71-131-114-113-113-115-99-97-103-137 71-137-103-97-99-115-113-113-114-131 97-103-137-71-131-114-113-113-115-99 97-99-115-113-113-114-131-71-137-103 99-115-113-113-114-131-71-137-103-97 99-97-103-137-71-131-114-113-113-115'
#sol = sorted(ok.split(' '))
#print sol
#
#print protein_translation('CCACGUACUGAAAUUAAC')
#print protein_translation('CCAAGAACAGAUAUCAAU')
#print protein_translation('CCUCGUACUGAUAUUAAU')
#print protein_translation('CCCAGGACUGAGAUCAAU')
