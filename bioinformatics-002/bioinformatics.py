# -*- coding: utf-8 -*-
"""
Created on Sun Nov 02 16:07:51 2014

@author: ngaude
"""

from collections import Counter
import operator
import math
import sys
import random

def ncr(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(operator.mul, xrange(n, n-r, -1))
    denom = reduce(operator.mul, xrange(1, r+1))
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


def min_hamming(pattern, text):
    '''
    returns the substring from text that
    minimize the Hamming distance with pattern.
    '''
    assert len(pattern) <= len(text)
    distance = sys.maxint
    minham = ''
    for i in range(len(text)-len(pattern)+1):
        s = text[i:i+len(pattern)]
        d = hamming_distance(pattern,s)
        if (d < distance):
            minham = s
            distance = d
    return minham

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

def all_kmers(k):
    '''
    return a list of all possible 2^k k-mers
    '''
    # iterate over all d-mer : from 0 to 4^k-1
    def itokmer(number,k):
        a = ['A', 'C', 'G', 'T']
        kmer = [None] * k
        for i in range(k):
            kmer[k-i-1] = a[number&3]
            number = number >> 2
        return ''.join(kmer)
    kmers = []
    for i in range(1 << 2*k):
        kmers.append(itokmer(i, k))
    return kmers

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
    if (type(peptide) == str):
        return peptide_mass_spectrum(peptide_masses(peptide), cyclic)
    else:
        return peptide_mass_spectrum(peptide, cyclic)


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


def leaderboard_trim(leaderboard, spectrum, N):
    '''
    output the N highest-scoring linear peptides on Leaderboard lp
    with respect to Spectrum
    '''
    # need for trimming ?
    if len(leaderboard)<=N:
        return leaderboard
    
    #build a dict of peptide:score
    d = {tuple(e): peptide_scoring(e, spectrum, cyclic = False) for e in leaderboard }
    ll = sorted(d, key=d.get, reverse = True)
    min_score = d[ll[N-1]]
    tp = []
    for e in ll:
        if (d[e]>=min_score):
            tp.append(list(e))
        else:
            # cut off loop optimization
            return tp
    return tp

def leaderboard_cyclopeptide_sequencing(spectrum, N, M = 20, convolution = False):
    '''
    find all peptides approximatively consistent 
    with a given spectrum
    '''
    lp = [[]]
    top = []
    top_score = 0
    spectrum.sort(reverse = True)
    lmass = None
    if not convolution:
        lmass = list(set(amino_acid_mass.values()))
    else:
        #build a specific alphabet from the convoluted spectrum
        candidate = [k for k in spectral_convolution(spectrum) if k>= 57 and k <= 200]
        dconv = Counter(candidate)
        lconv = sorted(dconv.items(), key=operator.itemgetter(1), reverse = True)
        min_freq = 0
        lmass = []
        for (k, v) in lconv:
            if (M > 0):
                lmass.append(k)
                min_freq = v
                M -= 1
            elif (M == 0) and (v == (min_freq)):
                #handle ties
                lmass.append(k)
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
                p_score = peptide_scoring(p, spectrum, cyclic = True)
                if p_score >= top_score:
                    top_score = p_score
                    top = p
            elif sum(p) > parent_mass:
                lp.remove(p)
        lp = leaderboard_trim(lp, spectrum, N)
    return top           

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


def neighbors(pattern,d):
    def mutation(prefix,suffix,m):
        ''' 
        return list of suffix-string 
        with at least m-mutation
        prefixed with prefix-string
        '''
        if (len(suffix)==0):
            return {prefix}
        res = set()
        if (m>0):
            nucleotides = {'A','C','T','G'}
            nucleotides.remove(suffix[0])
            for i in nucleotides:
                res |= mutation(prefix+i,suffix[1:],m-1)
        res |= mutation(prefix+suffix[0],suffix[1:],m)
        return res
    return mutation('',pattern,d)
    

def neighbors_2(pattern,d):
    '''
    returns a set of pattern which are at most at 
    hamming-distance d form pattern
    '''
    nuc = {'A','C','T','G'}
    if (d==0):
        return {pattern}
    if (len(pattern)==1):
        return nuc
    def immediate_neighbors(p):
        _nn = {p}
        for i in range(len(p)):
            mut = nuc.copy()
            mut.remove(p[i])
            prefix = p[0:i]
            suffix = p[i+1:]             
            for j in mut:
                _nn.add(prefix+j+suffix)
        return _nn
    def iterate_neighbors(d,nnn):
        if (d==0): 
            return nnn
        _nnn = nnn.copy()
        for _p in nnn:
            _nnn |= immediate_neighbors(_p)
        return iterate_neighbors(d-1,_nnn)
        
    return iterate_neighbors(d,{pattern})


def kmers(text, k):
    ks = list()
    for i in range(len(text) - k + 1):
        ks.append(text[i:i+k])
    return ks

def motif_enumeration(dna, k, d):
    '''
    brute force enumeration of (k,d)-motif 
    in a list of dna strings
    '''    

    patterns = set()
    dna_kmers = set()
    for s in dna:
        dna_kmers |= set(kmers(s,k))
    for pattern in dna_kmers:
        for neighbor in neighbors(pattern,d):
            is_candidate = True
            for s in dna:
                if (approx_pattern_count(s, neighbor, d)==0):
                    is_candidate = False
            if is_candidate:
                patterns.add(neighbor)
    return patterns

def consensus_motif(m):
    n = len(m[0])
    consensus = ''
    for i in range(n):
        prob = Counter([s[i] for s in m])
        sorted_prob = sorted(prob.items(), key=operator.itemgetter(1), reverse = True)
        consensus += sorted_prob[0][0]
    return consensus
    

def score_motif(m):
    n = len(m[0])
    score = 0
    for i in range(n):
        prob = Counter([s[i] for s in m])
        score += sum(sorted(list(prob.values()),reverse = True)[1:])
    return score

def score_motif_row_by_row(m):
    c = consensus_motif(m)
    return sum(map(lambda e : hamming_distance(c, e),m))

def entropy_motif(m):
    n = len(m[0])
    entropy = 0
    for i in range(n):
        prob = Counter([s[i] for s in m])
        for base, count in prob.items():
            prob[base] = float(count)/len(m)
        entropy += sum(map(lambda e: -e*math.log(e,2) ,prob.values()))  
    return entropy

 
def median_string(dna, k):
    dist = sys.maxint
    kmers = all_kmers(k)
    med = ''
    for pattern in kmers:
        d = 0
        for s in dna:
            d += hamming_distance(min_hamming(pattern,s),pattern)
        if (d <= dist):
            print 'median_string',d,pattern
            med = pattern
            dist = d
    return med

dna = ['CTCGATGAGTAGGAAAGTAGTTTCACTGGGCGAACCACCCCGGCGCTAATCCTAGTGCCC','GCAATCCTACCCGAGGCCACATATCAGTAGGAACTAGAACCACCACGGGTGGCTAGTTTC','GGTGTTGAACCACGGGGTTAGTTTCATCTATTGTAGGAATCGGCTTCAAATCCTACACAG']
        
print 'med',median_string(dna,7)
print '---------------------------'

def most_probable_kmer(text, k, profile):
    assert len(profile) == k*4
    dn = {'A':0,'C':1,'G':2,'T':3}
    most_prob = 0
    most_kmer = ''
    for kmer in kmers(text,k):
        p = float(1)
        for i, c in enumerate(kmer):
            p *= profile[i + k*dn[c]]
        if (most_prob < p or most_kmer == ''):
            '''
            note that the algorithm specifies that tie-break 
            should be resolved with the first occurence observed
            '''
            most_prob = p
            most_kmer = kmer
    return most_kmer

def form_profile(motif, pseudo_count = False):
    n = len(motif[0])
    probs = [0]*(n*4)
    dn = {'A':0,'C':1,'G':2,'T':3}
    for i in range(n):
        col = [s[i] for s in motif]
        row = len(motif)
        if pseudo_count:
            col += ['A','C','G','T']
            row += 4
        prob = Counter(col)
        for base, count in prob.items():
            prob[base] = float(count)/row
        for base in prob.keys():
            probs[i+n*dn[base]] = prob[base]
    return probs

def print_motif(m):
    k = len(m[0])
    profile = form_profile(m)
    for i,s in enumerate(m):
        print i,'\t'.join(s)
    for i,c in enumerate(('A','C','G','T')):
        print c,'\t'.join(map(lambda e: str(round(e,2)),profile[i*k:(i+1)*k]))
    return
    
def greedy_motif_search(dna, k, t, pseudo_count = True):
    assert len(dna) == t
    best_motifs = list([s[0:k] for s in dna])
    best_score = score_motif(best_motifs)
    motifs = ['']*t
    for motif in kmers(dna[0],k):
        motifs[0] = motif[:]
        for i in range(1,t):
            profile = form_profile(motifs[0:i], pseudo_count = pseudo_count)
            motifs[i] = most_probable_kmer(dna[i], k, profile)         
        score = score_motif(motifs)
        if (score < best_score):
            best_motifs = motifs[:]
            best_score = score
    return best_motifs

def randomized_motif_search(dna, k, t, iteration = 100):
    assert len(dna) == t
    n = len(dna[0])
    def randomized_motif_iteration():
        random.seed()
        r = [ random.randint(0,n-k) for i in range(t)]
        motifs = [dna[i][r[i]:r[i]+k] for i in range(t)]
        best_motifs = motifs[:]        
        best_score = score_motif(best_motifs)
        while True:
            profile = form_profile(motifs,pseudo_count = True)            
            motifs = [most_probable_kmer(s, k, profile) for s in dna]
            score = score_motif(motifs)
            if (score < best_score):
                best_motifs = motifs[:]
                best_score = score
            else:
                return (best_score,best_motifs)
    best = randomized_motif_iteration()
    for i in range(1,iteration):
        candidate = randomized_motif_iteration()
        if (candidate[0] < best[0]):
            best = candidate
            print 'iteration',i,'/',iteration
            print 'score :',candidate[0]
            print 'motif ',best[1]
        if (i%100 == 0):
            print 'iteration',i,'/',iteration
    return best[1]

def gibbs_sampler(dna, k, t, N = 100, iteration = 20):
    assert len(dna) == t
    n = len(dna[0])
    random.seed()
    def gibbs_sampler_iteration():
        r = [ random.randint(0,n-k) for i in range(t)]
        motifs = [dna[i][r[i]:r[i]+k] for i in range(t)]
        best_motifs = motifs[:]        
        best_score = score_motif(best_motifs)
        for j in range(N):
            i = random.randint(0,t-1)
            smotifs = motifs[:i]+motifs[i+1:]
            profile = form_profile(smotifs,pseudo_count = True)
            motifs[i] = most_probable_kmer(dna[i], k, profile)
            score = score_motif(motifs)
            if (score < best_score):
                best_motifs = motifs[:]
                best_score = score
        return best_score,best_motifs
    best = gibbs_sampler_iteration()
    for i in range(1,iteration):
        candidate = gibbs_sampler_iteration()
        if (candidate[0] < best[0]):
            best = candidate
            print 'iteration',i,'/',iteration
            print 'score :',candidate[0]
            print 'motif ',best[1]
        if (i%5 == 0):
            print 'iteration',i,'/',iteration
    return best[1]
       

assert greedy_motif_search(['GGCGTTCAGGCA','AAGAATCAGTCA','CAAGGAGTTCGC','CACGTCAATCAC','CAATAATATTCG'],3,5, pseudo_count = True) == ['TTC', 'ATC', 'TTC', 'ATC', 'TTC'] 
assert greedy_motif_search(['GGCGTTCAGGCA','AAGAATCAGTCA','CAAGGAGTTCGC','CACGTCAATCAC','CAATAATATTCG'],3,5, pseudo_count = False) == ['CAG', 'CAG', 'CAA', 'CAA', 'CAA']
assert form_profile(['AC','AA']) == [1.0, 0.5, 0, 0.5, 0, 0, 0, 0]
assert most_probable_kmer('ACCTGTTTATTGCCTAAGTTCCGAACAAACCCAATATAGCCCGAGGGCCT', 5, [0.2,0.2,0.3,0.2,0.3,0.4,0.3,0.1,0.5,0.1,0.3,0.3,0.5,0.2,0.4,0.1,0.2,0.1,0.1,0.2]) == 'CCGAG'
assert median_string(['AAATTGACGCAT','GACGACCACGTT','CGTCAGCGCCTG','GCTGAGCACCGG','AGTACGGGACAG'],3) in ('ACG','GAC')
assert motif_enumeration(['ATTTGGC', 'TGCCTTA', 'CGGTATC', 'GAAAATT'], 3, 1) == {'ATA', 'ATT', 'GTT', 'TTT'}
assert neighbors_2('ACG',1) == {'CCG','TCG','GCG','AAG','ATG','AGG','ACA','ACC','ACT','ACG'}
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
assert peptide_scoring('NQEL',[0, 99, 113, 114, 128, 227, 257, 299, 355, 356, 370, 371, 484], cyclic = False) == 8
assert peptide_scoring('NQEL', get_spectrum('NQEL'), cyclic = False) == 11
assert leaderboard_trim(['LAST', 'ALST', 'TLLT', 'TQAS'],[0, 71, 87, 101, 113, 158, 184, 188, 259, 271, 372],2) == [['L', 'A', 'S', 'T'], ['A', 'L', 'S', 'T']]

#
#dna = ['CGCCCCTCTCGGGGGTGTTCAGTAAACGGCCA',
#       'GGGCGAGGTATGTGTAAGTGCCAAGGTGCCAG',
#       'TAGTACCGAGACCGAAAGAAGTATACAGGCGT',
#       'TAGATCAAGTTTCAGGTGCACGTCGGTGAACC',
#       'AATCCACCAGCTCCACGTGCAATGTTGGCCTA']
#
#gs = gibbs_sampler(dna, 8, 5, 100, 20)
#print '\n'.join(gs)
#
#dna = ['GCGCACTACCTTTAATGCCCCTAGAGAGTCGAATTAGATACAGGCAGGGGCTTCGTTAGTCCATCAGCACTTAAAAACGCGCCAATTGGGACATCTCTGTGTCACTCGGCCCGGTGCCGGACGATAGTAACCTCGGATAAATCCGGCTATCCACGATTCGTATGGTATTCTTACTGGAATAGTGACGCTCCTCCTACCTTGGGCCTGGCAAAGGCATACCCTAGGATAGTTAAGACGTGTAAAACCACTCGACCGCTGTATAGGCGTAATAGCATGTCGATTTTAACAGTACTCTATAACAGCGCACTACCTTTAA',
#       'TGCCCCTAGAGAGTCGAATTAGATACAGGCAGGGGCTTCGTTAGTCCATCTCATATCGTTTAGTTAGCACTTAAAAACGCGCCAATTGGGACATCTCTGTGTCACTCGGCCCGGTGCCGGACGATAGTAACCTCGGATAAATCCGGCTATCCACGATTCGTATGGTATTCTTACTGGAATAGTGACGCTCCTCCTACCTTGGGCCTGGCAAAGGCATACCCTAGGATAGTTAAGACGTGTAAAACCACTCGACCGCTGTATAGGCGTAATAGCATGTCGATTTTAACAGTACTCTATAACAGCGCACTACCTTTAA',
#       'ACGAGATCGCGAATCTTCGGGATGGGTCCACCTCTTATTGGGCCTCGTGAAAGCCGAAATCACTACCTAAGGTCCATCTCGATCGGACCAAGGACATAGTTCCCGTTGCACCAACTCCAACGTGTCCTAGTCCTAATGCCACACGCCATCATCTGCCCGTATTCCTGCCTCTCTGCTCTTTGAGTTAGTGCCCTGCTGGCTTAAGTGGGGCTGCCATCAGTCCGTTTTAGTTGAATTCCGCAGGGTACGAAGCGTAAGGCTTATATTTTCAAGAATTTGAAAGACTGCCTACCACTTAATCGTGAATGCCGGTCTA',
#       'AAGTACCCTTCAAGTTGGCAAAACCAGGTGGCCGGGAGCGCACTTGGCTGTGGTGTGTTCCCAATAGTACTTATCCGTGTTCAGAGGACTACGCACTCGTATAGGCCTAACATAACTACAATACGTGCATGGGTGATGCAGGTGCTCATAAAAGGAGGGATGGGGACTATGCGCGACTAGGCGCCCTTAAGAGGACGCTCGACCTTAGACGCTAGTCCGTTTAGTCAACTAAACCGTATCCCTTGTTTTCGCGACCCACATCCGCTTCATGGGGTCAGCCCACCGTCTATATTACTTTAGAACGAGTCTCACTTTG',
#       'TCAGTAGCGACAAAGCTTTCAACTCGTTTAGTTACCAGCGGTAAATGCCCCTTCTCAGTCGTCAGATTACGTGATAGCCACAGTGTCCACTATAGCTATGGACATTCGAAGCAATGACCTTATGCAAGACTCATAGCCCCTGCTATCCACGTCAGCGTACCTATGGACATATTACGTGGAAGTTGCTCGAGGCATATCGCGAACGAGTGATTTTAGTGGCAGCGAGGTGTTCCACCCTGGTCTGGCCAATACCTTCCTTGATAAGAATTCCAGCGGAATACGTGGATTATAGGCACTAGCGTGTCAATCATATATA',
#       'GGAGATGAAGCTTAGCAGATAACCGAACGATTTATTGACTCATTTTAGTACGAGCGCGAAGCCTAACGAGGATATTTTGGGACGAGCTTATAGATATCTGTCTCATCAGTCTAGTTAGTTTTCGCAGAGCCGAGTTCGAGGTACAACAGACCGTTTGATGCCCTTCGAATCATGCATCGAATCCAATCTTCGCCCACGAAGCTCGCTCTATCATCCCCCGTATCGGACTAGCTTATATTAGGCCGGCTCGGGGGGTCACCGACTTCAGGTCTTCCTTTCAGCGCGTATGCACGGAGTTAGGCGAACACTTCCATAA',
#       'ACCGACCTTATTAGTGCATTGGACTAACTCAATCGCCCCTCCATATTATATATTTTGTCGCGCTGACTCTTCGATTGCGAATAAATGATCAAGCTTAGTGCGTTGTTTCCTAAGTCCGCCTACCGCATTGGAAGGACTTACCAGGATGTAAGACTGTGGACGCCCTAGACCAACCCCCTTGATTCTTCCGTTTAGTTATAGGATATCCCCCCGGAGGCAGCGAGGATTGCTCCCCCAACCAGTAGGCTAAAATACCGCCATGGACCACATGCGCTAATGAAAAGTTCTTTGAACCTTGACAAGTTATAGTATGTTA',
#       'AGATACCCAAACGTCGCCTGCCCTCAAACTTACGGTTCCTGAATGAGCCTGCCGGACCAGCGTTACTGATATAGTCTGCTGGAACCTTACTCAGAGCTTAGCGTAGGCCACCTGGCAGTGCAAGGGGTATGAGACCCCACTTCCTACATTGCCATGTATTTGGAAGCGAAGTAAATAGACTGTCGTGGGTAGCTCACCTCACTGCAACCTTTGATATCAGTCCGTTTATGGTGGCCGCACAGGTCCCCTCAGAACATATTTGAGACGGGACGTCTTAACAGATAGCGCCACAGAGCTCCGGCGAAAGAGTAAGCAA',
#       'TGTAGCTATCTATGGCCTAGACACTCCAGTGGTGCTGTTATGGTGACCCGTAGCACGCGTTAGCGTGATTACACGGTAGGTGCTCCGGGTACCCCTATTACGTTAGCTCTCCCATTCCGGTCTTGTATGGCGTTAATAGACACAGGAACAGTTGCAAAGACTTAAGGGTAGGCTCCAAAAAACATTTGAGGATGCGACGTATTAACATTTCAGTAAATGCAGGTTGTATCATTCATAGGAGTTCAGGAAGTTTAGTTCTCTATCCATACAGTGATTGAGGCCAAACCGATTATTTTAAGCCCACCATCAACGTATG',
#       'GGTTCGAGCCAGGCCCGCGTCGAGATAATCTCTATGGTTGTTGGCTCTGTATTCTTACAGTCCTGCGGCAGGTACAAGTAAGGATTAACCCAACGCTCGCTGTTTATACCCTCCTGGTGATTCGACCGCGCTGGGCTCGGGATCCCTGGAAACCCCGAGTCCCGAGACGCCTGACATCAGTGTCTTTAGTTCTGCATGCCTAAATCGCAGATTCGGCGCAGTTCGGTCTTAAGAACCGACCAGTATGATTTGCCTTCAAACTGGCCCTTGATCAACGCTTGTATGTTTGAAGTTGGATGTTTCGTGATACGCGCAG',
#       'TTCAATTGCATAAATGGCTAGGACACATCGTCGACTATTGTACGGGATTCCAGTCCGTTTAGAGGGCCTATGAACTTTGCGCGGTCACTATGACGGATTTACCTCCGCTATGCAGTGTAATTGATCTGGTGCTTTATCCCAGTCGGATGAGCGTGGGGGTGCAAACCCTTGTAACTAGTGCCTCACTCTAAAACCAATCGGCTCCCCTATGCCAATAACAACGGGCAATTTCGTATTTGTGTATACTCCCGTCTTGGTCACGGTTCTCGGTCGGAAAGGGCCAGTCCGATATAATAGATGGCGATTACCCGAGAGC',
#       'TGCCTCATGACTCTGGCCTTGTCAACTGTCCAAGCGGGAGGTTGGGACCCGCTAAATTTGGTCAACCTGCCTTTTAGCCACCTCAGTCTCTAAGGTATCAACAAATCTAAGACTTTGTTGCATGCGGTCAAAGGATTGCTCGTAGACCGCGCAGTGAGGGACCCTATCAGTCTAGTTAGTTAAAAAGCCACACATGGGTACATATCGGACGGTGCGGCACACTCAATGAGGGGTATGTACCGAAACCCACGACATCGATAGACTACCCACACTAGATGTATGTGGAAAACGTACCGATTCCAACCCATAGACAGAA',
#       'GTCAGTCCGTTGGATTACGGGAAGGAGACGTACGAGCAAAAGCGTTTGGCTTGAACCTTACAGACACGCATTTCTCCGACCGACTACGGTCAGGCTTTGATGGTACTTAGTGTCGATATACCTTGGACTCTCTTAGCGGCGTTCCTCCAGATTGTGCCCTATCCATAGCGCGCTCCCATTGCCGGTATTAGCGGCAGATGGTCTGGCTACATAAAGCCCAGTGGACACATTACGGAACGCGGGGGGCCATCGTGCTTCCAATTAACCTATGAGGTCTGATACGCCTCGCGATATTAGCGACGTCTTTCATTCGAAA',
#       'CAATTATCCGATACGAATAAAATAATTAACCTATCACAACGCAGGCCGGGTGAAGTGCGGTATAGCTTGTACTGTGGTGAAGTGTGCTGAAAAAGCGACCCTCCACGTGTCCGATCAGTCCCGCTAGTTTTGTCACACGCAATTGGAACCTAAGGAAATCTACACATAGGCGGAGAGCACTCTGTCAGTACTGCTGCGCAACCGCGGCGCGATGCTGAGAGGCCTATGATCCTACATGGAGCGTACGCATTACCCGTGACGTCACGACAATCCTAAGGCTTTTAGGGATAGTGCGGTTTTATCGTGATCTGGTAAT',
#       'TATCTTCAACTTGGAACGGATCATGAAGAAATAGGTTGGGTGTCATCATTTTCCTGGCGAGAGGTTCCACTGTACAAGGAGAATACTTTCAGAGTAATCTCTACTTTGCTGGTTCGATATCACAGCTACTACCGATGAGTACAGTAGTGTATTGATCCAGCCGAGCCATAGTTGGTGATACTCACCAGCTGTGCTGCCACAAGGATCCTGAAGTCAGACTCCTTCGTATGTAAGGCTGAAAGAGACATGTCTCGGCGGGGGGTCAGTAGCTTTAGTTCATCGCGTCATACAGTTATTCAGAATACCCAAGAGATTA',
#       'ACGACGAAACCCCCGGTCGATCACACCAAGAACATAGGGAGCAGGAGCTCTCCTCAGAACAGTCTCCTACTCGCAGTGTTCTCAATCCACGGCCCGCAGACAGCTTAGCGGAGTTAGGGCCCGCCCAGCTTCTCACGACTGCAAATGGCGGTTACCCTTCATGTACTTGGTCTAGTATCCTTCTTCCACGACGCTTGGTCAAATGTGGCGCGTATGTGCCTAAGTGTCAGTCCGAAGAGTTGCTATGCGTGAGTTCATCGCTAGCACTACATACCATCGAGACGAACGCCTATTTCACTCCATGATTTGTGGTTCA',
#       'GAAGGCGAAGTGTTAAATTTTCTGGTAACCCAACTAAATGGTTTGGGCATTTGAGCGCTTCGACGACTTTATGGATTTACGTTCCATGTTACGCTCTTGACTGGGACCGCAATCCGGCACATAAGCGTGAGGAATGCTTCATTGAGTGTTCGTCGTCAGTCCGTCAGGTTATCCCTCACCATCATACGGATACACTTTCTTTTTTAGACTAATGAAAATATTGGGACTTTCCCTATGGACGCTTTAACACTTGGTCGGGCTTGGGGGCACAACCGCCGGACTGCGATGAGCTTAACGGGCACACATTGACGTGCCC',
#       'TTGTCTTCGTTGATCAGGGCTTGTTGAGCGACTCTAACCGAGCAATGAAATGAACTTGAGGGGGTCATCCGATTATGATCATCAAGGAAGACGAGCTACGCAAGTACACTGGAGTAGCAACATGGACACGAGTTTCGTGCCGTTTAGTTGAGGTGAGGTTTTCAGGATTGATATTCGTACCACTGATACCGGCAACCTCTCCAATACGACTCTCGCCACGGTAGTTGCAGAGTCGGATCCGCCAGGGGGGGAGAAGTTTCTAGCCATTAAGTCAGTGCTTACCAGAGGGGGCTATTTCTTAGCACGGCACAGATCG',
#       'TTAAATTTAGCATAGAGGGTGCTGTGTACATAGCCCGCAATAGGAAGACAATTGTCCGACATCACGCGGCATTTCAGTGATCTGACGGAGTTCTCACCCGATATTAGCGCTTCGCTCTGTCAGACTAGACGAGACTGGGGCAGGACCCGCGATAACGTCCACCGGTATGATATCCGGGTCCAAGAGCGCCATATGACACTTTGTTCCGCGTCCCAATAATATGACTTGTTTTGTACGCGTGGCACACCTCGACGCCGCGTAGCTTTGGCTCAGATGGGGGCAGGTCCGTTTAGTTTTGTCGCAAAGTTTTGGACAT',
#       'GGGACCCACCTCCGCCATTATGTCCAATGCAAATCAGAAGCTGTCGATCGGATTGTTTGTTACCACCTGGTGAGGTTTCACCTTTAATAGTATACCCATAATCTACTAGAGGCCAGCGTGGGCGCCTAAGCTGCGACTTATAACGTTCTGTGGGCGGCGAACAGAAGCACGTGTGACCTTAGAACCAGAGCTCGGGTTTCTCTGGCACCTTGCGTCTATCTTAGTCGTCGATTGCCGACTTGATGTACCTCCTGTCACGGCCGTTCCTGGACCTTGTCAACCCGAGAAAAGCCATCAGCAAGTTTAGTTTCCCAAA']
#
#gs = gibbs_sampler(dna, 15, 20, 2000, 20)
#print '\n'.join(gs)
#
#
#
#dna = ['GGACGCAAGGTTAGTCATACTTCTGTTTCACGCAAGCTAGTGGCTATCGCAGACTTAATCGCCAGGTAGTAGCCGGCCGTTGCCGATTTCCTGGTAACAGGCTGTTATCGATTATCTTTTCGGGCTCGTGGACACAGGGGAGCGAGCGGGATGAGCTGAGGGACGCAAGGTTAGT',
#    'CATACTTCTGTTTCACGCAAGCTAGTGGCTATCGCAGACTTAATCGCCAGGTAGTAGCCGGCCGTTGCCGATTTCCTGGTAACAGGCTGTTATCGATTATCTTTTCGGGCTCGTGGACACAGGGGAGCGGGCCTCCGTTTTCCGAGCGGGATGAGCTGAGGGACGCAAGGTTAGT',
#    'GAGATATGAGATGCGGATTTCCACGCCAGAGTGTTCCGTTTTGTACACTAAATGTGCCCTACGCACGCTAGGTTCCCCGCGTGATCTTTTTGGGTAACGAGTACGTGGGTTAATAATAATTAGGGTGGTATCCACAACCTGCCTCCTGATTCGTCGCTTATCAAATTTATGTCAG',
#    'ATATGTCCACGACACGACATTGGTCCATAGCTGTTCAACCTCCCGCTCCGGATCGCTACAAAGACTGTCTATCACAAGTATTGCAAGAATCTGAACGAATCCCAGCTAGGGCAACAAGCCCTGCCGCGTTCGGACCTCTTTCAACGGTGTTCCACCTTCAGCCGATTCACGAGCA',
#    'CTGCGTCCTTCCGTTTTCCATTGCCGACTTTTTGCTATTCCCAACCACTGACAAAGACGATGGAAAAGGTTAACGGGTTACGATCGTGCACCAGCTTGCACCGTTGAAGAGTTGCTTAGCACCTGTTTTGATCCGATGGTGGTGACTTAATTATCACACCAATCACTGGCAGGCA',
#    'CCCTGCGCGGACCGTCATAATTTTCGGCGATGATGAGGCGCCCACACGAAATGTTAACAAGCTACAAATGCTCATGTGTTCCAGAGTCGTGATAAATCAAGGTGTTTAGCCGGTGTTCCGCAATCTATGAAATGTCCGGGAAGATAGTGTTTTAGTCGAATGGCCCGGACCTTTA',
#    'AAGTCCGTTCCGACCTTATTCTACTTGGGCTGTCGTAGAGCGTTCAGCTTCCAGAACCTCTCGGTTCGCGCGTCAACATATGCGAGTAAGGACCACTGGGTACATCCTTGCGCTGCTACGGTGTAGTGTTTTCCCTTCTTTGTAATGAATGAGGGTGTACAGTTATGACCCGCAC',
#    'TTGATGAGCGAACCCACATCGGCCACACATCCTGTAATAGTCATTCAGTCGAATGATTCAGGGGATGATAGTTCGAGGTAAACATCACTACACGGGTGTGAGGTTCTACTAGGGGCGTGTGCGGTGTTTTTTTTTCTAGACTCCAAAGGGCTAGCCTCGTTGCCTCTGGTGGCGC',
#    'TAAGAGCGTAGCTACGCAGTTACTTGGCCTCTGACTGGTATCTAGCCGTCCGGTGTTGGGTCCTCAAAGACGTAAGAGCTTCACGGACTTAAGGATAACTCGAGGATTAAACTGCTACAGGGCCTACAGGAGCATCATTCCGGTAAGCCGTTTTCGGATTTGTGGCACGGCCTAT',
#    'GCTTGAGACTGAGTCGGATGCTGTGTAAGTTCAGGCATTCTACATGCGTCGTCCATGATGCGATCGATAGCAAAACCGTTTCGTGCAGAAACCAAAGCGCCCGGGAGTTTCGGATGAACTTCGGTGTTCACATTTCTTTCTGTTCCTAGTTTTAGGCACGGTGAGTGGCTAACAG',
#    'AAAAGGGTCCTGCGGGTCTCGTTATGGAGTCGATGACCGGTTAACCGTTTTCTGACCGGTAACCAGTTAGGGGAGCATCCGAGCCAGGGAGTTATCCTGTCTTTCACCCGACCGTCCAATCCGGAATATGCTAGCGCCGATAGTCGTGTACGCACGCCAAGGCTTGTAAACGTAG',
#    'TCGTGCAAGCGTACCCCTCTCTCACGTCCGAAATTATCTGTACGGATCAGATTCGTCGGAAGACGGTCTAAATGTTCCGTTTTCCTGTGACAAGACGAACATGCCTCATAACCATATGTCTAAGAGGATGATATTAGGTCGAATCCCCGATGGGCGGTGAAGGCTGATATCCCAG',
#    'TAACGTTCGTCCAATGGAGCGCCGGCTGAGTCCGCCTCCCGGCTTACTGGACCCACTACATGACGCTTGTCGCCGGTCTGCAAGTGTACTCGGAACTCCGTTTTCACAGCCGGCGAGGCCAGTCAGCGATAACCAATGCCATATCGAAGTTACTACCAAGCCCTCGTGTAACCAT',
#    'GCTGGTCTGGGTATTTAAACGTGCGGCCGCAGGGCTAGCAAATTCTGGAAGATTCTGGTAGCATATCACGGATTCCGGTGTTCCGTTACGCGGACGTGTCGGACCCTCGGGACGCGAAAGTCCAGTGTTTGTCCCCACTCGGTCTACACTATTTGGGAGCATTAATGTGGGGCCC',
#    'CTTGTCTGCCCATGCCAATGAAAACGAATACCCACAGCTTTAATAATCTAAACGCGTGTCATACCAAGTCGGCCCTGAAGCCCGGTGTGGCACATCTCACTTTTCCACAGAGTATTCGTCCGGTGCATCGTTTTCCGAGCACACCACTGTATGCTGCTCAATCGTGAACGATCGC',
#    'GAGTAGGGGACGCAGGAAAATTTCGTTCATGAAGGGAGGAGATACTCTTCTAATTATGCTCAAAACATCGCCACTAGTTATGCTCTAACTTTAGCCCCAGCGCAGTTGCAACAATCAGGTCAATTATTCGCGATAGCTAGAAGAGTAAGGGGAGGTGTTCCGTTTGTCCAACTGG',
#    'ATTCCACCTGTTTCATCTAATGGAACATCGGGTCGACCGCTATTTCGGGATAAACTATTGTAAGGCAGCGTACGGTGTATGGTTTTCTATGCCCTAAAAAGGATGACAAAGTCAGTTTAAGCTATATAGTCTCTCAAAATAGCGTCAAATCTAGGCCATCCTGTCAGAGTGGGCT',
#    'CACCATATAAACACTCGGTGTTCCGTCAGCGCTCACGTGGCCGGGGCTATCATCGCGCCTTGAGGGAAATGTGCTCATGGCGGCTACCCTAATAGGCATCTCCTTGGATGCGCGGGCGGAGTTTATAATCACCTCGAGTAAAGTACTAAAACTCTATTAAGCGCGTGTATCAGAG',
#    'ACGGGTTCTTTGACATCCTCGCACGTTATACCCGCTAGTAATTAGATAGGTAGAACCGAAATGACTGGGTAGCCGGTGGGTCGTTTTCGCGCGGTCCGCAGGACGTGGAGTAGGACTTCGGAACCGGTTTGTATAGGTGGCGCACTCTAGAGTGTCCAAAGATTAACCGACTTTA',
#    'GTTTATCTTACCACTTGGACTTGCATTGAGAGATCGGTCTTATACCGCAATTTTCCTTCGGATCGCAAAGTCAGTACGTCAGGGTATACAAACTCTTGGTTCCGTTTTCTTAGCACGAGTCCTACACGTGACCAATGCGGCCGGTCCTGACTCAGTTGATGAATAAATAGTAGGC']
#
#rm = randomized_motif_search(dna, 15, 20, 500)
#print '-------------------------------------'
#print '\n'.join(rm)
#
#            
#dna = ['CGCCCCTCTCGGGGGTGTTCAGTAAACGGCCA',
#       'GGGCGAGGTATGTGTAAGTGCCAAGGTGCCAG',
#       'TAGTACCGAGACCGAAAGAAGTATACAGGCGT',
#       'TAGATCAAGTTTCAGGTGCACGTCGGTGAACC',
#       'AATCCACCAGCTCCACGTGCAATGTTGGCCTA']
#rm = randomized_motif_search(dna, 8, 5, 100)
#print rm == ['TCTCGGGG', 'CCAAGGTG', 'TACAGGCG', 'TTCAGGTG', 'TCCACGTG']
#
#dna = ['AAGCAGTTATATATCTTATCCAACTCGCCGCGCATGGGGATGATGCGGTGAGACATTCGACTCGACTCAGTTGACGTGTCATTCACAGAGCTATACTTCAAGCGTCGAACATTTCCTTATACTGCACCTTCATTGAAAGCTCGAGCACCGTACAAT',
#       'ACGCCTCGTTGGCTGTGTTTGGATGTAATACAATGTGAACTCAAAGGTCGATAACGCTAGGCTATATCAAGGCTCAAATTGCAGTGCTCGTATGTTGAGCACACGTGGTGTGGTTTCGGTTCGACGCTCATGCTAATATGTGTTACATAGCACTCG',
#       'TCAAGGCATTCAGGGTGATACCCATTATGCTGCCATTAGAGACATCTGGCGGCTGTGCACTCACCAGTATATCCGTGGGTGTCTTCTTCGCCCATGGGTTCGGTCCAATCTTGCTTACTCGACAAATGTGGATTGGACCGTCGAATACCAGTAAAA',
#       'ATGGCAAGTATTGTCAGCAACTCTCGATTTGCAACGCACCCTCGAGGTAGTCGCGAACCGGGTTCGCAATACGGAGTACTAGACCTATGTCACTAGTACAGGTCTCGTCTCTAGATAGGCTCATCGCACATGTGTTATTCACAGTCCCCTCGGGGA',
#       'GGCGATGACCCCTAGATTGAGGAGTGTGCGGTGGCCCTCAAACCGTAAACTTGAAAGCTCTCCGTCAAAAACTCTTCGCTCCTGCGGGAGGGACTAGCAGTTTCCCAGCTAGTGAAGAGGTAATTAGACGTCTGCGTTTCCTAGCGAGTAAGCGGT',
#       'ATGTTCGTTGCAGGTCTCACGCTTAAGTACCTGGCGCCAAGGTTCTACCAGGATCGACTATCGACTCAAAGTTGCCCGCGTCGATCAGTGTGGAAGATCTTTGTTATTTCTACGCGCTTGGGCCTGGGTTTTCTAGCAACGCAGACTTAATAACGG',
#       'CCGAGAAGCTGAATGTAACTATCCCTGAAATACGCAACGGGGCAACCATCTGCGCGCGTGCTTTACTAGAGGTTGAGCTAGAATAATAAAGTAGAACGGTCGAATCCCGGGCACGAGTTTCCGCATCCTTAATCTTAACAAATGTAACAAGCGATT',
#       'CCCCCGGAGATGTCTTCGCGCGTGCTTTGGGGTAGTGCTCAGTTTTAGGCAGCGATTAATCGGTCGTGAGGATAGGTGCCGCAGCACGCCCTTTTAGAGCTACCCACTTTGGTGGTTCACCGGCGAGGCTGTGTCGGACCTGGCGTCTCGTTGACT',
#       'CAACTCTTCATATCCAGCCGTCATCATGTGGGGGAAGAGCTAAAGTGGACGTGATATCCCATTTGACCGCTAGGCGCAAAGAATCCGTCTCTAAGGCCGAGTATCCCCTCATCGCTCTTGTTTTGACCCCACGTCCTGGAGAACGCATGTAATTTA',
#       'GACGTATTTACCATGGCTGAATGGTTTCCGATGAATAGGATAACACCACTAGGCCAGCGAACGATGGCTAACTGCCCTTGCTCTTCCCCGCACCTGTGGTGCTGCAGTTCCGTCCTTCCCAGGGAGCAATTGGGATTAACTCAATGCGGGTTTTTA',
#       'CCTAATTGAAATGACGAAATCTACTTTACTCCATAGGATCGAAAAAAGATAGGCCTGTTCCCGGGCACATTGTCCCCGCACGTGGATTCAAGAGAAAGTGGTATCCACCTGAAACAACTGGGGAGGTTCGGACTGTGCCTCACGCCCCCACGGCAG',
#       'CCAAACATGTCTTCAGCGCTCATGTCATGTTCCCACATCATCGTTCGGGATATTCACACATCTTCTACCAAACCTCGCATGATTATCCCTTAGGTCGCTGTCTTGAACCAAGATTGACTCGGAGAGTCTTCAAAAGAGATCTAACCGTGTCATCTT',
#       'CATAGCCGCCCGGTCCATGATAGACCGATGTCAGCGAACAACCACAATACATCACTATAAAAGAAGTTAGCAATGTCCAAAGCTACCACGGATTCTAACTATACATACATATGAGAGCACTCACCGCTCTTGGCTACGCACTGAGCGTGATATACA',
#       'AAGGTCCCGACTAATCATCTACGTCCAGTAACTTTTGATTCTTCGGCAAGTTATCGTCACACTTTATTGCTTGATCACGAGTCAGTGCAATGGCAGTCTGCGCTCATGAGCGAAAGCTCTTCTAGAAGGCAGCGATACGTGATCGTGTCATCAATA',
#       'GTGCGATATATAGACCAGTTTAATAAGGGGGCCGCAACCCCGCCAGCTCCTGGGGTTAAATCATGGACGAGTTAGCGAGTTACCATCTATCCTGATTAATTTGTGTTTAGAAGTCGTGGGAGTAGCGTGACATCATCGCGCCTGCCCCGTCGTTGA',
#       'TACCGGAGCACCTACCCCCTCCGGTCAACTTCCATCTCGCCGCCCATGAGCCGTCAATCCACGACTCCCGAGAATTCTTGCGGTTGGAGGCAGACGACCATTGGTTACCATATCTGCAAACGCCGAAGTTTAATAGCTAGGCCTCGTACAGCGTAA',
#       'CACGACGTTAATCTCGCGGGGACTTCTTCGCGCCTGAAAAGCACTGAGTCACACATCTTACTTTGGTAGTTCAATCCATGTGTACTCGGATAGCTCGATAGTACAGCGCATGCAGGCGTTTTGCTCGCTCGAGCCCGAGCTCGGACTTGAATTCAC',
#       'GAACCTCAGATATCGGCGCGTGAGATATCGAATTTGATGAAAACTAATCGCACCGACGGCGAGGTAAGAAGGCTGCAGTGGAGCCCAGCGATACCACTCCGGAAAGATGGGCGCAATATTTCCTCGCCCTTGGTATCCAAAGTGCCTGACCAACCC',
#       'CGTCGTATTGCCCTTCACACGCTCGACCGAAATCCCTCGTGACGCTGCAGTACATCTCTTGGTGCTGCCTGGAGTACTTAGGACTCTACGTTGGAGACTAGGATGCGATTTCTTCAAAAGTCGCTATGCCGGTCGACGCTCATGGCTCATTCGTAA',
#       'GAAACCGTTGGTTATACTGCCTACCCAGCCAGTACATTCGGGAGGGTGTATCTTGCGTAGAGAGACATCTGGGGTCAGTTCGGAAGGTGGATGCTCGATTTAGCATCGGGGTAACACTACGTAGATCCTCGGCTTCGGGATATCTCTCCGCGCTTG',
#       'CGACCTGTTCTTATAGGGATCCGCTCTTCTGCAACTGTGGTTATGCGGGGGAGGCTACTCTCCGCGCACTTGGGATTGGACAGTACTATCCTTGGTCCTTGGCAAGGGCCGATCTTTACCTACAGCCTATTTGCGGGTAATCCCCTAAGGCCGTCC',
#       'TCTGCGGGGCGCCGAGTAATTGGGCCATAACAGCAAGCGCCTGACTTTTAAGTATGGTTCCGCAAAAGTGAGTCTGATTCTTCCTGAGTACCTTCCTCTTCGCGCATGGTCGATCTCCGCGTCTGGATGCAAGCTTAACTGGCGACTCAAACACAG',
#       'GTGGCCATTCGTCAAGGACTCAATTCAACGCCCTTGCCTTATGTTATCCATATCATCAATTTAGTCACACAAAACCGCTCCCCGAAGACGTCCACTAGACGTGAACTGGTCATCAGCGTCGGCCTTTCCCCTGGCTTACACTAGGAGGAAACCCAC',
#       'CAGGCGTGGAGAACAGCAAACTTCATGCGTACTGCACTACACTGTGCAGCCACTATTCGAAGGGTTAAATCTTCGTCGCGCTTGCTCGCCTCACCAGCGCATCGAGGCGCCCTTGCTACTACCCTTTATAGCCGACGCACCCTGCGAATAGATGGC',
#       'GCCTCCCCTCGGTACTTGCGGGTCTCGCCGCACGTGGTAGTCTCCGCATCGAGCCATCTCGAAGCCAGTTGATAGACAGCGTCCGGTATTTAGGCGGAAGATTGGTGTCACCGCTCAATTGACCCTTGCCCGATAGCTGATTACAATCACCCCAGA']
#bm = greedy_motif_search(dna,12,25)
#
#print ' '.join(bm) 
 
#
#dna = ['CCTGCGTCATCTGCGGTCTTCGTCTTATGTAGCGGCGCGACTATTGCGCTCTCAAGAGAGGTTCCTGGGCGTCTCTCATCTGATGGTTCACCTATTGGCACCCTTACAAGGTTTGGATAAAGAACAATGCAGCAACAAGATTATAATCCCCGAACG',
#       'GTCTCTTTCGGTCCATTACTATCCCTCTGAAGGGAGTCCCCTGTTATTCAGTCTACCGCTGACATCGGGTTTCCTTAGGTTCCATACTATTGAAAGAATTCCATCATGGGGCCAATCTCAGTATGTTGCACACCTCAACCCACATTTCGTATCACG',
#       'TTGGTAAATAGATTGTAGAGTTATAACCGAGCTCTCTGCTCGTAAAAATGGACGCACTAACCCTCGAGGGAGTCTGGATCCCGGTTAGTTATTTGGTCTCACTTATTGGTTATCAAGCGACCTTTCGAGAACCCCTTGTCAACATCACGTGGCTGT',
#       'ACTGGCCTTTGCCAGCTTGATCCCACGAGCTGTCTTATTAGCTTCGACGCCCCCGAATCCGAAATATCAGGACGCTACAGTGAGTGCAGGAAACGAAAACACGAAGCTAAAGGTGTTGGAGCCTATCTCTACTATGAGCAAAGAGGCACACACATA',
#       'CAGTAAGACCATGCTCCCCTCACAAAGGTGAAAGATAACAATATTAGGCCCTACAGCGAGGGGTCAAGTGGGTGGCATGGGCCTGCCAAAGACAGCAATATTGGAGTAACTAGTTACGATTGCGTGCTAAAACCTCCTACATGATCTAACTATTTA',
#       'GGTAAACTCTGAGTTGTTGAGTTCTATTAACGTTCAAAACATACCCAACGGGGAAAATCCCACTTTAGAGAGCGTCCGCGGGCCTGATTGTCTGCGTCCCTTAGTATAACTACATGACCCTTGGATCAGGACGGACATCTTTCCCGTCGAGACTCA',
#       'CTCCACCTATTATACTCTTGCTGGGGCCACTGGACCCATTTGCAGCCAGTGTCTACCAGATCTCTGCGTCTCACAAGCAGGGCTTGTTGCTAAGGCTATTTATAAGGCCCCTGGAGTGAGCGGTCTTGTGAACGTTACCGTAGCGCATTCGGCTTG',
#       'TTACGCGGTATGGGGACGGCGACGGGTGGTGAGTTCTACTCTTGGACGAACAGATCAGGTCCCCACATGTCAGGAAGATCAGAGTGACGCACTCGGAAGCTTTGGATTTGAGTCCTCCCTCCGTGAGCGTGTGCGTGGTTAGTACGCTCTAGTGAG',
#       'ACGCTCTTCCGAGGGGTGTTTGGCCATGCTAACCCACCCTTCAGAGAGTTAAGTTGTAAGCTTGTCGCCCGAACCACACCCTTTGCCGTATTCCAAGTACTAATTCAGGGCTAGTAAGGTTCCTCGTCCAGAATCCGGAACGTGCACAGCGAATTA',
#       'TATACTGACACCGCAGGGGACGGAAGTTTAGGCGTAAGCTAGTGGAATGTCACATTCATCCCGATTCATGGTCGCTTAAGAGAGTTCTAAGCCCCTCGAAGGAGCCTATCAGTCAAAAATCAGTCGCCCCAGTATCATTGTGCGGAGCTGTTTTTA',
#       'ACGGTCGAAGACTCTCTAGGCACGCTCTAAAGGGAGGGCTTCATTGGTAGTAAACGATTCTAGATTCATGGTGATCGAGATCCGCTTGTCCAACTTATTGATTGTATATCATGGCAGGGAGCATCAGACGTGTTTTGAACCACATCCGGAATCATC',
#       'AGCCGCACTTGGTACAATTCTGACAATCATACAACGCGTCCCGGATTGGTAGGCTAACGAGTCTGGACCGATCGGAGGACATCTCCAGATCCTAGAAGAGGCGCGACCAGCCGTGATATATTCTTCTTTCTTCCCTTTAGGGAGATAGTAATTATT',
#       'TTCGCTTGTTTACGCTATAGTGAGACCTGTAGCTTGTCACTGTAAGTGTACGGGTATAGGGATTGGGACGCCACGTAACCGAAACGTATTGGGATGACTAAAAATACTTTCCAGTTAGTGCTCCTATTCTGGACCTGAGTTAAGCGAAGATCGCTA',
#       'TGGTATGCTCGCGCGCCAAGGAAGCCGGGATTTCGAAGGTAGTGCACGCGGGCGTTCCGAAATGGGCCTAACTAGTTCTCTTACTATTCCGAGTAAAACAGATCAGGCGTTGGGAACAGTCGCTTAAGAGAGAAGCACCCCGTTCTTCTCCAAATC',
#       'CTGACTATTGCGGTACGTTTGGGTGACTGCCAACCGACTCGACCCCTCGGCCCTGCTTTCTGTTTTTCCTTCGACTGCGGCAATCCACGTACCTCGCACTGAAGTGAGGTGTTCATACCGCAAACAAAATTCTGAGGAAAGGAGGCTTATCGTTGA',
#       'CCGAACTCACAGCGGTCTCCCTGGATAAACAAGAGCCAGGGTCCACATAAGCATATTTAGTCATCTCGCCAATGCGTAGCCATTCACACCTGACCTCCCTATAGAGAGTTACACCAAAGAGAGGCCAACCGGTTTCAAAACAAACGAGTTCGTGGC',
#       'GTTCATAAGGAAAAATTGAACGACTTATGCCTTCCCGCAAGCACTCAGCTCTGCAGGGAGGAAAAAAGTCTATGGGATACAGGGTCTAAGCACCTATGCCTCAGGAACCTCATCTAAGTGGCACCCGACCACAAACATCTATACGATGTCTCCTAG',
#       'GGATTATGGGCTTAGCTACTTATGGCGGTGCGCAGGTATAGATGAATCGCGGCAAGACAATAGCACAGTATCGAAAGACTATTGCTCTAGAGCGAGTTGGCCAATAACAATGTCTGTCTTAGGTTGCTGAGATTCTCCAAAAGTTATGAACTTGTC',
#       'CGCGGAATCATTCTGGAGTAGTGGGTTAGGAAACAACTATGGGGACTAAAATAAGGAACACGCTATAGTGAGTACACAGACAGTCAAGGGCATAAGAATGCCGTACTGGGTGCCTGTATATGTATGCGCACGCAAATATCGTCCTCACTAGGGTGA',
#       'CGCTGAAGAGAGCTCTATAGCACTCCATGCGTGCCTGTATATACCCACTTCGATAAGCATCGGAGCCCCTGGCCGGGACAGTTTCTCTTGCCAGTAGCATATCATCCCCGGAACTTAGCACGGCGGATCAGAGTATAATGGTCCGTCTTCACGTCC',
#       'CTGTGGAAGGTTTCGACGCTGGTTAAGGTCCGCGACCCCATCTTTGGTACTGGTTACAGTCTATCTACGGGGGCCAGCGAGCGTGGCGTTGTATGTAGGGCGGACCCTCCCTTAAGCGAGTCGACGGCTGCAATTGCAAAGGGACTCACTTTAACA',
#       'AATTGCAGGTAGCGCTTGAGTGAGCTTAGTTCGTACTGCCGGGCTCTGCCTAGACTCTCAATCCCTAGAGCGATACGGGCCTATAAATCTAACCACAGAGTTGGTCCTGGGGGACATGGCCGAATGATTGTGCGAGCAGCCCAACAATAAGCTGAG',
#       'CGCCGGGGTGCAGTCTAGATAACAGTGCGACGCTCTTGGCATAGAAGCGCCCACTCCGAAATCGGATCAAAGCCTTACGACTGGCTTACCACAGTCAAGTGTATATAACGTTGAACTTGATGTTACAATGGCGTTTACTACGAGCGCTCCAGCGAG',
#       'GCCCAATGCAGAAAATCAGCAATCCCCTGTAGCGAGTCGGACTATCTCAAGGACGCAGTATGAAGTGCGCTTAGCGGTATAATCCTCTGGCTAGCCAATATGAACGCCCAGCTGTCCCACAGATGTGACTGGTATCATTTCTCTTGACGACGACTC',
#       'TCCCTGCTTGTCATCATAGCATAATAACTAAAAGCAAAAACGCGAATCGGCAAGAAGGTTGCCTAGGCTAGACGGGGAGGACAACACTAAAGGGAGACAGACGGATGAACCAGGGGAAGTATGTATCCCACCGATTAGGGAACCGCCGGTAAGTGT']
#bm = greedy_motif_search(dna,12,25, pseudo_count = False)
#print ' '.join(bm)
       
#    
#
#prof = [0.2,0.2,0.3,0.2,0.3,0.4,0.3,0.1,0.5,0.1,0.3,0.3,0.5,0.2,0.4,0.1,0.2,0.1,0.1,0.2]
#text = 'ACCTGTTTATTGCCTAAGTTCCGAACAAACCCAATATAGCCCGAGGGCCT'
#print most_probable_kmer(text, 5, prof)
#
#prof = [0.364,0.333,0.303,0.212,0.121,0.242,
#0.182,0.182,0.212,0.303,0.182,0.303,
#0.121,0.303,0.182,0.273,0.333,0.303,
#0.333,0.182,0.303,0.212,0.364,0.152]
#text = 'TGCCCGAGCTATCTTATGCGCATCGCATGCGGACCCTTCCCTAGGCTTGTCGCAAGCCATTATCCTGGGCGCTAGTTGCGCGAGTATTGTCAGACCTGATGACGCTGTAAGCTAGCGTGTTCAGCGGCGCGCAATGAGCGGTTTAGATCACAGAATCCTTTGGCGTATTCCTATCCGTTACATCACCTTCCTCACCCCTA'
#print most_probable_kmer(text, 6, prof)
#
#
#prof = [0.379,0.273,0.182,0.227,0.227,0.288,0.333,0.273,0.379,0.273,0.197,0.273,0.121,0.197,0.258,
#        0.167,0.379,0.258,0.288,0.258,0.227,0.242,0.197,0.227,0.242,0.242,0.197,0.242,0.303,0.212,
#        0.212,0.136,0.303,0.212,0.273,0.258,0.242,0.288,0.167,0.212,0.258,0.227,0.379,0.258,0.227,
#        0.242,0.212,0.258,0.273,0.242,0.227,0.182,0.242,0.227,0.273,0.303,0.303,0.258,0.242,0.303]
#text = 'GTAAGTAAGCAGTGTCGGGATCTACGATCTTAAGCCTCCCGCTGCGAACACTAGTTTCTTAGCGAGTTTTCTCCTGATATCTTGAACCCGCAGAGCTCTGTTATGTACTTGGGCGTGTCGAGCTGAGAGTGTCATTACGTGTCGAAAAAGCCTGTTTCTACATTCCCCGAGCGAGTGTAGGGCTGTAAATTGCTAGAAATACCGCGGTGTTGTAGCCACCGATAACCAGTGGAGCAAAGGACAAGTTCTCGGCTTCCGTGGCTTATTAGCGAAGCACACCCGTGGGTACCAGAAACTTATTTAGCAGGGGAACCCGAGGTTCTTTGACTGAGGTCAACGATGTCGGGCGTGACGTACGCTGTTGGCACCACGTCGCGTAGTGATTTAGAAGGTCGGAGGTTGCTCCGTATTAGTGGTACGCTCAGGTTGGCACACCGCTGGCCAACGCCCTCAAGAATCGGAATTCAGGTTTTCTAGTCTGAGTAGTACAGCAGCGATTCCCGGAGGTAAAAGATTCAACCGGGCTCTTACATGCTAGAACCCAGTCTCGATCGCTATTTCACTGCTCATCTGGCCCTTGACAGGGAATCGGTGCACCGATTCCCCAATTCAGCCCTCACACAGAGTGACTTCTGACATGATTACTCCTCAGTCTCTATCGCACATGCTCATCGCTCAGCTATTCTTTTGTGAAAAGGCTTCAGCCTCGCTGCGGTGATCCTACCTCGATTCTACTAACGCGTCTCGAGGGCATACGGGCTTACCACGAACTCCCAAGTACCATTATGGTCTCAACTGATTTGGTAACCGAACACTCCCTAAGTGTCCTGTACATTATAAAAAGGAACGTCACCGCCATTCTTATTAGCACAAGTGTTGGAATTGAGGGGCCGCTCCGCTAGCAAGCCCGCAGTATTCCAACTTGGCTCAACTCGAGCAGTCACAAAACATGGTAGACCGTAATTATTGTATGTTGGGACTGTCCTAGAGGCCCAAGTCC'
#print most_probable_kmer(text, 15, prof)

#
#dna = ['TGATGATAACGTGACGGGACTCAGCGGCGATGAAGGATGAGT',
#       'CAGCGACAGACAATTTCAATAATATCCGCGGTAAGCGGCGTA',
#       'TGCAGAGGTTGGTAACGCCGGCGACTCGGAGAGCTTTTCGCT',
#       'TTTGTCATGAACTCAGATACCATAGAGCACCGGCGAGACTCA',
#       'ACTGGGACTTCACATTAGGTTGAACCGCGAGCCAGGTGGGTG',
#       'TTGCGGACGGGATACTCAATAACTAAGGTAGTTCAGCTGCGA',
#       'TGGGAGGACACACATTTTCTTACCTCTTCCCAGCGAGATGGC',
#       'GAAAAAACCTATAAAGTCCACTCTTTGCGGCGGCGAGCCATA',
#       'CCACGTCCGTTACTCCGTCGCCGTCAGCGATAATGGGATGAG',
#       'CCAAAGCTGCGAAATAACCATACTCTGCTCAGGAGCCCGATG']
#print median_string(dna,6),'vs CGGCGA'


#dna = ['CCTGGAAATCGTCATTGCTCACCTCAGCCATAGTCCGTTCTC',
#       'CGTGCGTGTTTCCACGAACATTCGCTGCCATCGGTCCCGTTC',
#       'CTTCAGCGGCCAACCCAATACGTACGTGGGATAAACTTCCAA',
#       'TCACGCCTCTGGCGGCCAGTACGCACCGATTGATGCTCATAG',
#       'TGAGCATTGATCAGAAAGCTGCCAGCGTGAGGCACATGTGAT',
#       'GCTGGATCCAATTGGTAGGCCAAACCGCCAGGACATAGAAAC',
#       'GGAGATCGTAACCTTGAAAATCCGCCGCCACCGTAGAGTCGC',
#       'TACCTGCCCTACGTTTTTCAGGTTCCAGGGCGGCCAGCCTGG',
#       'CTCTTAAACTGCCTACGCCTGCCATAGATGCAGACTCAGAAA',
#       'TGGTGACAGCTGCGGCCACTTCCCCGTGCCGTAGCCAACACT']
#print median_string(dna,6)




#
#motif_matrix = ['TCGGGGGTTTTT',
#                'CCGGTGACTTAC',
#                'ACGGGGATTTTC',
#                'TTGGGGACTTTT',
#                'AAGGGGACTTCC',
#                'TTGGGGACTTCC',
#                'TCGGGGATTCAT',
#                'TCGGGGATTCCT',
#                'TAGGGGAACTAC',
#                'TCGGGTATAACC']
#  
#print 'entropy',entropy_motif(motif_matrix)  
#print score_motif(motif_matrix)
#print score_motif_row_by_row(motif_matrix)
#print consensus_motif(motif_matrix)
#

#k = 5 
#d = 2
#dna = ['GCACGCGCCATCGGTTGTCAAGGGA',
#       'TTTATTAGACAAACTACGCTCGTTT',
#       'AGAAGCTCCCGCTGTGACAGGCGAT',
#       'TGTCGCCGGTGTAGATGGTGAATTG',
#       'TTTAATTAACACAGATCGTTCGTAG',
#       'CTTTCACGTTTCGGTACACGATATC']
#motifs = motif_enumeration(dna,k,d)
#print ' '.join(sorted(list(motifs)))


#M = 20
#N = 1000
#s = '0 97 99 113 114 115 128 128 147 147 163 186 227 241 242 244 244 256 260 261 262 283 291 309 330 333 340 347 385 388 389 390 390 405 435 447 485 487 503 504 518 544 552 575 577 584 599 608 631 632 650 651 653 672 690 691 717 738 745 770 779 804 818 819 827 835 837 875 892 892 917 932 932 933 934 965 982 989 1039 1060 1062 1078 1080 1081 1095 1136 1159 1175 1175 1194 1194 1208 1209 1223 1322'
#ls = map(int,s.split(' '))
#ld = leaderboard_cyclopeptide_sequencing(ls, N, M = M, convolution = True)
#print '-'.join(map(str,ld))


#M = 20
#N = 60
#s = '57 57 71 99 129 137 170 186 194 208 228 265 285 299 307 323 356 364 394 422 493'
#ls = map(int,s.split(' '))
#print leaderboard_cyclopeptide_sequencing(ls, N, M = M, convolution = True)

#
#N = 1000
#s = '0 97 99 113 114 115 128 128 147 147 163 186 227 241 242 244 244 256 260 261 262 283 291 309 330 333 340 347 385 388 389 390 390 405 435 447 485 487 503 504 518 544 552 575 577 584 599 608 631 632 650 651 653 672 690 691 717 738 745 770 779 804 818 819 827 835 837 875 892 892 917 932 932 933 934 965 982 989 1039 1060 1062 1078 1080 1081 1095 1136 1159 1175 1175 1194 1194 1208 1209 1223 1322'

#ls = map(int,s.split(' '))
#ld = leaderboard_cyclopeptide_sequencing(ls, N, convolution = True)
#print '-'.join(map(str,ld))

#
#N = 364
#s = '0 71 87 87 97 97 101 101 113 114 129 131 131 137 137 137 137 137 147 147 158 163 163 184 198 208 211 224 224 234 238 244 245 248 260 264 266 274 274 285 292 294 294 294 295 295 309 321 335 342 348 353 361 371 391 393 395 396 406 407 407 408 411 411 422 422 424 425 429 432 458 479 485 493 493 498 505 508 508 516 530 533 538 538 539 540 554 558 559 559 561 580 587 595 595 616 617 622 630 630 636 639 642 652 662 671 677 685 687 688 692 696 696 701 709 717 717 724 749 753 759 764 767 773 779 793 799 800 802 802 806 824 825 825 832 833 835 840 846 854 854 886 896 904 907 910 911 916 922 926 930 931 933 937 937 946 965 972 982 983 991 991 1001 1003 1009 1023 1023 1024 1033 1038 1043 1044 1047 1047 1068 1073 1088 1094 1095 1096 1100 1110 1115 1119 1120 1128 1134 1144 1146 1155 1160 1170 1170 1175 1180 1201 1202 1210 1220 1225 1225 1226 1231 1231 1232 1246 1247 1247 1257 1257 1281 1283 1307 1307 1317 1317 1318 1332 1333 1333 1338 1339 1339 1344 1354 1362 1363 1384 1389 1394 1394 1404 1409 1418 1420 1430 1436 1444 1445 1449 1454 1464 1468 1469 1470 1476 1491 1496 1517 1517 1520 1521 1526 1531 1540 1541 1541 1555 1561 1563 1573 1573 1581 1582 1592 1599 1618 1627 1627 1631 1633 1634 1638 1642 1648 1653 1654 1657 1660 1668 1678 1710 1710 1718 1724 1729 1731 1732 1739 1739 1740 1758 1762 1762 1764 1765 1771 1785 1791 1797 1800 1805 1811 1815 1840 1847 1847 1855 1863 1868 1868 1872 1876 1877 1879 1887 1893 1902 1912 1925 1928 1934 1934 1942 1947 1948 1969 1969 1977 1984 2003 2005 2005 2006 2010 2024 2025 2026 2026 2031 2034 2048 2056 2056 2059 2066 2071 2071 2079 2085 2106 2132 2135 2139 2140 2142 2142 2153 2153 2156 2157 2157 2158 2168 2169 2171 2173 2193 2203 2211 2216 2222 2229 2243 2255 2269 2269 2270 2270 2270 2272 2279 2290 2290 2298 2300 2304 2316 2319 2320 2326 2330 2340 2340 2353 2356 2366 2380 2401 2401 2406 2417 2417 2427 2427 2427 2427 2427 2433 2433 2435 2450 2451 2463 2463 2467 2467 2477 2477 2493 2564'
#ls = map(int,s.split(' '))
#ld = leaderboard_cyclopeptide_sequencing(ls, N)
#print '-'.join(map(str,ld))

#
#N = 10
#s = '0 71 113 129 147 200 218 260 313 331 347 389 460'
#ls = map(int,s.split(' '))
#
#print leaderboard_cyclopeptide_sequencing(ls, N)
#
#
#N = 325
#s = '0 71 71 71 87 97 97 99 101 103 113 113 114 115 128 128 129 137 147 163 163 170 184 184 186 186 190 211 215 226 226 229 231 238 241 244 246 257 257 276 277 278 299 300 312 316 317 318 318 323 328 340 343 344 347 349 356 366 370 373 374 391 401 414 414 415 419 427 427 431 437 441 446 453 462 462 462 470 472 502 503 503 511 515 529 530 533 533 540 543 547 556 559 569 574 575 584 590 600 600 604 612 616 617 630 640 640 643 646 648 660 671 683 684 687 693 703 703 719 719 719 729 730 731 737 740 741 745 747 754 774 780 784 790 797 800 806 818 826 827 832 833 838 846 846 847 850 868 869 877 884 889 893 897 903 908 913 917 930 940 947 956 960 960 961 964 965 966 983 983 985 1002 1009 1010 1011 1021 1031 1031 1036 1053 1054 1058 1059 1062 1063 1074 1076 1084 1092 1103 1113 1122 1124 1130 1133 1134 1145 1146 1146 1149 1150 1155 1156 1171 1173 1174 1187 1191 1193 1200 1212 1221 1233 1240 1242 1246 1259 1260 1262 1277 1278 1283 1284 1287 1287 1288 1299 1300 1303 1309 1311 1320 1330 1341 1349 1357 1359 1370 1371 1374 1375 1379 1380 1397 1402 1402 1412 1422 1423 1424 1431 1448 1450 1450 1467 1468 1469 1472 1473 1473 1477 1486 1493 1503 1516 1520 1525 1530 1536 1540 1544 1549 1556 1564 1565 1583 1586 1587 1587 1595 1600 1601 1606 1607 1615 1627 1633 1636 1643 1649 1653 1659 1679 1686 1688 1692 1693 1696 1702 1703 1704 1714 1714 1714 1730 1730 1740 1746 1749 1750 1762 1773 1785 1787 1790 1793 1793 1803 1816 1817 1821 1829 1833 1833 1843 1849 1858 1859 1864 1877 1886 1890 1893 1900 1900 1903 1904 1918 1922 1930 1930 1931 1961 1963 1971 1971 1971 1980 1987 1992 1996 2002 2006 2006 2014 2018 2019 2019 2032 2042 2059 2060 2063 2067 2077 2084 2086 2089 2090 2093 2105 2110 2115 2115 2116 2117 2121 2133 2134 2155 2156 2157 2176 2176 2187 2189 2192 2195 2202 2204 2207 2207 2218 2222 2243 2247 2247 2249 2249 2263 2270 2270 2286 2296 2304 2305 2305 2318 2319 2320 2320 2330 2332 2334 2336 2336 2346 2362 2362 2362 2433'
#ls = map(int,s.split(' '))
#
#print leaderboard_cyclopeptide_sequencing(ls, N)


#p = 'HWYCTVKLVMFCLPIPSVAPMWLYTGRIFTANVDGIGAMFG CTFSPKMVHDCEMSMFDQKWAWPMNDQDALEIWKVTHKDVA SRQANCCPENVKWMMITVFPGDLFNSRPCCKTNNSYVHMEP RGKSRVPCERLCTVPRAVTTQREPGCLESYTGIKSCLEVND FLYELLYINDPRWHHRAGNAIIWWLKLLRSYWQTSYIRDIT MEGPMCLALLCFDSGWYYGKGYRCVLTYRMVGRGKPIKYRG LYWFAHKVCCAHSCAMDRPIWHVDQEKNFKMFARPDAHAMY DDRMLHSDATQQFEVLQTFYTGLCCPGQHCPNREPYSEKAL QPTEPRATSDQDERSYYMWSVACMYLLHSGKEYNGNGNAGN VIHFFVICCHDQNGNYDFEKDQYCDLIHCSLLVEMYLRPKP FWFPQMNKITIESIIIAKKVAFYTQGDTLFMDQIEDYYCYC CCQGSWSWYEMYINCPGTVHKFGWQTSPYRDAGQGWGHSHS DYLTSEQLNFRHHHIYNSLFGVEQYHYTECHPLDELNEAKF PWHGYLMVDLKNAEIECFVSHRYDWIWDCYRCRMFAQRYHQ NSWMSEVNEPSKITQGPQEAFKKRPEKFDQPNRMEILYHDW IKQYLMVDINDQKEEGHSHRSKLMFGWCQKATLCLGGHEAK QYCSRRMVLGIHGWIWCENEYGSWADYPCVYVRTGPRNTQI'
#s = '0 57 71 71 87 97 97 97 97 99 101 101 101 103 103 103 113 113 113 113 113 128 128 128 128 129 129 131 131 131 137 137 137 147 147 156 156 156 156 163 163 168 184 184 186 194 199 200 200 200 210 213 213 216 225 226 230 230 234 234 234 238 242 248 248 250 255 256 259 259 259 260 262 266 269 274 275 276 281 284 285 287 297 297 303 310 312 313 316 323 327 329 331 331 337 338 339 341 349 351 352 357 363 369 371 372 377 383 385 386 390 390 390 397 398 403 410 410 410 413 415 416 418 422 423 425 430 438 440 447 459 460 460 468 472 472 474 478 478 480 481 486 493 494 494 495 497 498 500 503 511 513 514 515 521 523 523 538 546 546 550 551 557 559 560 566 566 572 577 579 587 590 591 594 597 597 599 600 603 603 608 609 615 616 628 631 634 637 641 649 652 654 656 656 658 660 661 661 669 678 679 679 688 694 702 703 706 713 716 721 722 728 728 731 731 734 737 750 750 750 753 753 753 755 757 759 759 762 762 763 789 789 797 800 805 808 810 815 816 817 821 829 831 834 834 849 850 850 854 862 862 862 863 864 865 869 878 884 886 887 890 891 900 907 909 915 916 917 918 918 920 928 928 931 934 936 962 965 966 968 975 983 987 987 988 992 993 997 997 1000 1001 1001 1005 1006 1017 1018 1018 1019 1031 1037 1037 1038 1044 1047 1051 1059 1059 1062 1063 1064 1071 1084 1091 1094 1096 1101 1114 1118 1120 1121 1124 1128 1131 1134 1135 1138 1138 1139 1146 1149 1150 1153 1154 1160 1160 1165 1166 1168 1172 1175 1181 1187 1191 1200 1215 1218 1225 1231 1234 1235 1236 1238 1247 1248 1249 1251 1251 1252 1252 1253 1259 1267 1269 1272 1273 1275 1277 1294 1297 1310 1316 1319 1321 1322 1328 1331 1332 1337 1343 1349 1350 1354 1355 1356 1365 1365 1366 1367 1372 1380 1388 1394 1395 1398 1400 1409 1410 1414 1415 1416 1420 1424 1428 1429 1434 1452 1453 1459 1465 1466 1466 1468 1469 1478 1481 1483 1485 1494 1495 1497 1510 1511 1512 1517 1518 1523 1531 1545 1549 1551 1552 1555 1556 1561 1565 1565 1566 1567 1579 1582 1584 1590 1596 1598 1609 1615 1622 1628 1631 1632 1639 1646 1646 1648 1650 1651 1658 1659 1662 1667 1668 1678 1680 1689 1696 1698 1712 1718 1723 1727 1728 1729 1731 1735 1740 1745 1745 1746 1747 1749 1750 1751 1759 1765 1775 1782 1790 1795 1797 1814 1815 1817 1825 1826 1831 1832 1846 1848 1849 1851 1852 1864 1866 1876 1881 1883 1887 1888 1888 1892 1893 1908 1912 1913 1915 1923 1928 1928 1935 1944 1953 1954 1962 1969 1975 1979 1980 1982 1986 1988 1989 2001 2011 2012 2020 2025 2025 2025 2032 2044 2045 2049 2051 2055 2056 2069 2072 2081 2082 2088 2099 2100 2111 2113 2122 2122 2125 2138 2142 2145 2148 2153 2158 2161 2162 2168 2169 2172 2182 2183 2186 2201 2209 2212 2219 2228 2235 2238 2242 2244 2245 2250 2259 2269 2271 2274 2285 2286 2289 2295 2298 2299 2306 2306 2311 2329 2332 2338 2343 2359 2372 2372 2382 2384 2387 2392 2396 2398 2401 2403 2403 2411 2414 2417 2430 2435 2451 2458 2460 2466 2471 2485 2485 2485 2490 2495 2499 2500 2501 2520 2524 2532 2545 2548 2548 2566 2567 2572 2579 2582 2584 2591 2597 2598 2602 2605 2614 2614 2616 2629 2645 2646 2648 2663 2669 2676 2681 2685 2701 2704 2710 2710 2715 2719 2728 2729 2733 2742 2742 2747 2749 2761 2766 2782 2792 2800 2801 2802 2804 2809 2814 2832 2841 2850 2861 2862 2873 2875 2884 2889 2889 2895 2897 2898 2903 2905 2910 2911 2915 2932 2949 2963 2987 2988 2997 3000 3004 3008 3012 3017 3017 3020 3026 3039 3045 3050 3059 3061 3062 3088 3100 3100 3116 3129 3139 3140 3145 3148 3151 3156 3159 3160 3163 3164 3187 3201 3213 3213 3216 3242 3260 3263 3272 3276 3279 3285 3287 3295 3300 3300 3307 3329 3342 3347 3369 3376 3388 3397 3398 3398 3400 3407 3410 3413 3435 3455 3460 3463 3478 3497 3498 3510 3523 3525 3531 3538 3554 3563 3576 3591 3610 3611 3611 3626 3628 3634 3638 3651 3662 3666 3673 3723 3737 3738 3739 3747 3757 3759 3767 3779 3801 3818 3820 3850 3850 3851 3866 3870 3888 3895 3915 3921 3937 3948 3948 3963 3979 4001 4018 4026 4034 4044 4050 4050 4076 4076 4121 4131 4147 4147 4157 4157 4163 4204 4218 4234 4260 4260 4260 4275 4313 4331 4331 4347 4373 4388 4416 4428 4459 4460 4460 4529 4556 4557 4573 4616 4670 4685 4729 4798 4826 4954'
#N = 5
#lp = p.split(' ')
#ls = map(int,s.split(' '))
#print ' '.join(leaderboard_trim(lp,ls,5))


#p = 'WFEKVEYEHAENVRCSFVTPYQITHHNQWGTTNSN'
#s = '0 71 71 87 97 99 99 99 101 101 101 101 103 113 114 114 114 114 128 128 129 129 129 129 137 137 147 147 147 156 156 163 163 172 186 186 186 198 200 200 200 201 201 202 213 215 227 227 228 234 238 242 243 246 250 250 255 257 259 260 266 266 292 292 294 297 301 302 309 314 314 314 315 316 333 333 342 342 347 351 355 356 358 358 361 363 363 363 364 365 369 389 393 395 397 403 406 413 413 416 422 428 429 438 443 444 460 460 461 464 465 470 472 472 477 490 492 492 493 494 494 496 498 500 505 517 519 536 541 542 544 551 553 558 559 560 561 561 569 571 576 584 589 591 593 597 601 601 606 607 619 622 629 629 647 652 652 657 658 660 662 664 665 672 672 673 678 679 685 689 694 697 698 705 714 728 732 736 743 745 746 748 751 753 754 758 761 766 779 785 786 786 793 798 799 807 808 811 819 826 828 834 835 842 850 850 852 856 857 857 859 861 865 872 873 875 883 895 898 899 900 908 912 925 935 935 949 954 955 956 964 966 966 971 971 971 979 982 985 986 987 987 994 998 1006 1011 1012 1013 1013 1026 1036 1036 1042 1045 1053 1055 1057 1063 1065 1070 1095 1099 1100 1101 1101 1107 1111 1112 1118 1125 1127 1129 1135 1137 1141 1142 1142 1150 1154 1158 1158 1166 1171 1189 1192 1192 1198 1208 1213 1214 1226 1226 1228 1230 1236 1237 1239 1241 1248 1251 1253 1258 1263 1272 1285 1295 1298 1304 1305 1312 1314 1318 1322 1329 1329 1337 1338 1340 1351 1354 1355 1355 1357 1375 1377 1384 1395 1395 1399 1400 1408 1409 1413 1419 1426 1426 1426 1432 1439 1451 1451 1452 1457 1458 1476 1484 1494 1496 1496 1500 1504 1522 1523 1524 1527 1531 1538 1540 1547 1550 1555 1555 1555 1564 1586 1589 1595 1595 1597 1597 1604 1614 1618 1623 1623 1626 1643 1650 1652 1656 1663 1664 1678 1687 1692 1696 1698 1701 1711 1717 1718 1718 1722 1724 1727 1733 1742 1751 1777 1777 1789 1790 1790 1793 1797 1798 1806 1812 1815 1821 1823 1836 1843 1847 1850 1855 1864 1873 1889 1890 1891 1899 1905 1906 1911 1912 1918 1920 1937 1937 1944 1951 1976 1977 1984 1984 1990 1992 1992 1998 2013 2019 2019 2020 2027 2036 2048 2055 2058 2083 2084 2091 2091 2091 2093 2112 2113 2123 2137 2140 2145 2148 2148 2156 2156 2183 2184 2194 2205 2205 2211 2212 2219 2234 2247 2249 2254 2259 2269 2270 2283 2284 2285 2292 2293 2308 2334 2340 2348 2350 2361 2369 2381 2382 2383 2384 2395 2397 2405 2406 2406 2411 2422 2449 2462 2464 2470 2490 2509 2511 2512 2520 2521 2526 2535 2544 2551 2561 2563 2563 2567 2568 2591 2597 2634 2648 2649 2649 2650 2662 2665 2673 2677 2692 2697 2698 2724 2730 2744 2748 2762 2763 2764 2764 2777 2806 2825 2834 2835 2845 2853 2859 2876 2876 2877 2878 2893 2926 2930 2948 2954 2963 2964 2982 2990 3004 3007 3031 3040 3055 3062 3062 3078 3091 3095 3119 3127 3168 3169 3190 3190 3192 3209 3218 3220 3241 3256 3281 3306 3319 3321 3337 3346 3370 3376 3393 3395 3420 3435 3447 3507 3522 3523 3523 3532 3534 3548 3621 3633 3636 3662 3679 3709 3734 3735 3749 3780 3848 3863 3865 3881 3935 3966 3995 4049 4067 4082 4181 4196 4268 4382'
#ss = map(int,s.split(' '))
#print peptide_scoring(p, ss, cyclic = False)

#p = '97-129-97-147-99-71-186-71-113-163-115-71-113-128-103-87-128-101-137-163-114'
#s = '0 71 71 71 87 97 97 99 101 103 113 113 114 115 128 128 129 137 147 163 163 170 184 184 186 186 190 211 215 226 226 229 231 238 241 244 246 257 257 276 277 278 299 300 312 316 317 318 318 323 328 340 343 344 347 349 356 366 370 373 374 391 401 414 414 415 419 427 427 431 437 441 446 453 462 462 462 470 472 502 503 503 511 515 529 530 533 533 540 543 547 556 559 569 574 575 584 590 600 600 604 612 616 617 630 640 640 643 646 648 660 671 683 684 687 693 703 703 719 719 719 729 730 731 737 740 741 745 747 754 774 780 784 790 797 800 806 818 826 827 832 833 838 846 846 847 850 868 869 877 884 889 893 897 903 908 913 917 930 940 947 956 960 960 961 964 965 966 983 983 985 1002 1009 1010 1011 1021 1031 1031 1036 1053 1054 1058 1059 1062 1063 1074 1076 1084 1092 1103 1113 1122 1124 1130 1133 1134 1145 1146 1146 1149 1150 1155 1156 1171 1173 1174 1187 1191 1193 1200 1212 1221 1233 1240 1242 1246 1259 1260 1262 1277 1278 1283 1284 1287 1287 1288 1299 1300 1303 1309 1311 1320 1330 1341 1349 1357 1359 1370 1371 1374 1375 1379 1380 1397 1402 1402 1412 1422 1423 1424 1431 1448 1450 1450 1467 1468 1469 1472 1473 1473 1477 1486 1493 1503 1516 1520 1525 1530 1536 1540 1544 1549 1556 1564 1565 1583 1586 1587 1587 1595 1600 1601 1606 1607 1615 1627 1633 1636 1643 1649 1653 1659 1679 1686 1688 1692 1693 1696 1702 1703 1704 1714 1714 1714 1730 1730 1740 1746 1749 1750 1762 1773 1785 1787 1790 1793 1793 1803 1816 1817 1821 1829 1833 1833 1843 1849 1858 1859 1864 1877 1886 1890 1893 1900 1900 1903 1904 1918 1922 1930 1930 1931 1961 1963 1971 1971 1971 1980 1987 1992 1996 2002 2006 2006 2014 2018 2019 2019 2032 2042 2059 2060 2063 2067 2077 2084 2086 2089 2090 2093 2105 2110 2115 2115 2116 2117 2121 2133 2134 2155 2156 2157 2176 2176 2187 2189 2192 2195 2202 2204 2207 2207 2218 2222 2243 2247 2247 2249 2249 2263 2270 2270 2286 2296 2304 2305 2305 2318 2319 2320 2320 2330 2332 2334 2336 2336 2346 2362 2362 2362 2433'
#lp = map(int,p.split('-'))
#ls = map(int,s.split(' '))
#print peptide_scoring(lp, ls, cyclic = False)

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
