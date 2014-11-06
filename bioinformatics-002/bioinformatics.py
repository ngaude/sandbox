# -*- coding: utf-8 -*-
"""
Created on Sun Nov 02 16:07:51 2014

@author: ngaude
"""

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
   
def linear_spectrum(peptide):
    '''
    Generate the theoretical spectrum of a cyclic peptide.
    '''
    global amino_acid_mass
    peptide_mass = sum(amino_acid_mass[i] for i in peptide)
    s = [0,peptide_mass]
    for i in range(len(peptide)-1):
        for j in range(i+1,len(peptide)):
            subpeptide_mass = sum(amino_acid_mass[k] for k in peptide[i:j])
            s.append(subpeptide_mass)
            s.append(peptide_mass - subpeptide_mass)
    return sorted(s)

test = '0 71 71 99 101 103 113 113 114 128 128 131 147 163 170 172 184 199 215 227 227 231 244 259 260 266 271 286 298 298 310 312 328 330 330 372 385 391 394 399 399 399 401 413 423 426 443 443 470 493 498 502 513 519 526 527 541 554 556 557 564 569 590 598 616 626 640 654 657 658 665 670 682 697 697 703 711 729 729 753 753 771 779 785 785 800 812 817 824 825 828 842 856 866 884 892 913 918 925 926 928 941 955 956 963 969 980 984 989 1012 1039 1039 1056 1059 1069 1081 1083 1083 1083 1088 1091 1097 1110 1152 1152 1154 1170 1172 1184 1184 1196 1211 1216 1222 1223 1238 1251 1255 1255 1267 1283 1298 1310 1312 1319 1335 1351 1354 1354 1368 1369 1369 1379 1381 1383 1411 1411 1482'
res = ' '.join(map(str,linear_spectrum('IAQMLFYCKVATN')))


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
assert ' '.join(map(str,linear_spectrum('LEQN'))) == '0 113 114 128 129 227 242 242 257 355 356 370 371 484'

print ' '.join(map(str,linear_spectrum('GWHANHDMQGIP')))
