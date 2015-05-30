# -*- coding: utf-8 -*-
"""
Created on Sat May 30 23:59:23 2015

@author: ngaude
"""

import numpy as np


def hmm_path_prob(path,states,transition_matrix):
    """
    CODE CHALLENGE: Solve the Probability of a Hidden Path Problem.
    Given: A hidden path π followed by the states States and transition matrix Transition of an HMM
    (Σ, States, Transition, Emission).
    Return: The probability of this path, Pr(π).
    """
    #print '>>',[(states[i],states[i+1]) for i in range(len(path)-1)]
    return 0.5 * np.exp(sum([np.log(transition_matrix[states[path[i]],states[path[i+1]]]) for i in range(len(path)-1)]))

def parse_path_states_transition_matrix(text):
    lines = text.split('\n')
    path = lines[0]
    states = {s: i for (i,s) in enumerate(lines[2].split(' '))}
    transition_matrix = np.zeros((len(states),len(states)))
    for i in range(len(states)):
        transition_matrix[i,:] = map(float,lines[i+5].split('\t')[1:len(states)+1])
    return path,states,transition_matrix

text = 'ABABBBAAAA\n--------\nA B\n--------\n	A	B\nA	0.377	0.623\nB	0.26	0.74\n'
assert np.abs(hmm_path_prob(*parse_path_states_transition_matrix(text)) - 0.000384928691755) < 0.000000000000001

text = 'BBABBBABBAABABABBBAABBBBAAABABABAAAABBBBBAABBABABB\n--------\nA B\n--------\n	A	B\nA	0.863	0.137\nB	0.511	0.489\n'
assert np.abs(hmm_path_prob(*parse_path_states_transition_matrix(text)) - 3.26233331904e-21) < 0.000000000000001

def hmm_emission_prob(emission,symbols,path,states,emission_matrix):
    """
    CODE CHALLENGE: Solve the Probability of an Outcome Given a Hidden Path Problem.
    Input: A string x, followed by the alphabet from which x was constructed, followed by
    a hidden path π, followed by the states States and emission matrix Emission of an HMM
    (Σ, States, Transition, Emission).
    Output: The conditional probability Pr(x|π) that x will be emitted given that the HMM
    follows the hidden path π.
    """
    assert len(emission) == len(path)
    return np.exp(sum([np.log(emission_matrix[states[path[i]],symbols[emission[i]]]) for i in range(len(path))]))

def parse_emission_symbols_path_states_emission_matrix(text):
    lines = text.split('\n')
    emission = lines[0]
    symbols = {s: i for (i,s) in enumerate(lines[2].split(' '))}
    path = lines[4]
    states = {s: i for (i,s) in enumerate(lines[6].split(' '))}
    emission_matrix = np.zeros((len(states),len(symbols)))
    for i in range(len(states)):
        emission_matrix[i,:] = map(float,lines[i+9].split('\t')[1:len(symbols)+1])
    return emission,symbols,path,states,emission_matrix

text = 'zzzyxyyzzx\n--------\nx y z\n--------\nBAAAAAAAAA\n--------\nA B\n--------\n	x	y	z\nA	0.176	0.596	0.228\nB	0.225	0.572	0.203\n'
assert np.abs(hmm_emission_prob(*parse_emission_symbols_path_states_emission_matrix(text)) - 3.59748954746e-06) < 0.000000000000001 

text = 'zyyyxzxzyyzxyxxyyzyzzxyxyxxxxzxzxzxxzyzzzzyyxzxxxy\n--------\nx y z\n--------\nBAABBAABAABAAABAABBABBAAABBBABBAAAABAAAABBAAABABAA\n--------\nA B\n--------\n	x	y	z\nA	0.093	0.581	0.325	\nB	0.77	0.21	0.02\n'
assert np.abs(hmm_emission_prob(*parse_emission_symbols_path_states_emission_matrix(text)) - 3.42316482177e-35) < 0.000000000000001 


############################################################
fpath = 'C:/Users/ngaude/Downloads/'
#fpath = '/home/ngaude/Downloads/'
#fpath = 'C:/Users/Utilisateur/Downloads/'
############################################################

#fname = fpath + 'dataset_11594_2.txt'
#with open(fname, "r") as f:
#    print hmm_path_prob(*parse_path_states_transition_matrix(f.read()))
    
fname = fpath + 'dataset_11594_4.txt'
with open(fname, "r") as f:
    print hmm_emission_prob(*parse_emission_symbols_path_states_emission_matrix(f.read()))