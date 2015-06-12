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
    return 1./len(states) * np.exp(sum([np.log(transition_matrix[states[path[i]],states[path[i+1]]]) for i in range(len(path)-1)]))

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


def hmm_decoding(emission,symbols,states,transition_matrix,emission_matrix):
    """
    CODE CHALLENGE: Implement the Viterbi algorithm solving the Decoding Problem.
    Input: A string x, followed by the alphabet from which x was constructed,
    followed by the states States, transition matrix Transition, and emission matrix
    Emission of an HMM (Σ, States, Transition, Emission).
    Output: A path that maximizes the (unconditional) probability Pr(x, π) over all possible paths π.
    """
    assert transition_matrix.shape == (len(states),len(states))
    assert emission_matrix.shape == (len(states),len(symbols))
    rstates = {v: k for (k,v) in states.iteritems()}
    def __log_weight(l,k,i):
        si = symbols[emission[i]]
        return np.log(emission_matrix[k,si]*transition_matrix[l,k])
    score = np.empty(shape = (len(states),len(emission)), dtype = float)
    backt = np.zeros(shape = (len(states),len(emission)), dtype = int)
    
    score[:,0] = np.log(1./len(states)*emission_matrix[:,symbols[emission[0]]])
    for i in range(1,len(emission)):
        for k in range(len(states)):
            pscore = np.array(map(lambda l:score[l,i-1]+__log_weight(l,k,i), range(len(states))))
            score[k,i] = pscore.max()
            backt[k,i] = pscore.argmax()
    # backtracking max score from backt pointers
    rpath = []
    state = score[:,len(emission)-1].argmax()
    rpath.append(rstates[state])
    for i in range(1,len(emission))[::-1]:
        state  = backt[state,i]
        rpath.append(rstates[state])       
    return ''.join(rpath[::-1])


def parse_emission_symbols_states_transition_matrix_emission_matrix(text):
    lines = text.split('\n')
    emission = lines[0]
    symbols = {s: i for (i,s) in enumerate(lines[2].split(' '))}
    states = {s: i for (i,s) in enumerate(lines[4].split(' '))}
    transition_matrix = np.zeros((len(states),len(states)), dtype = np.longfloat)
    for i in range(len(states)):
        transition_matrix[i,:] = map(np.longfloat,lines[i+7].split('\t')[1:len(states)+1])
    emission_matrix = np.zeros((len(states),len(symbols)), dtype = np.longfloat)
    for i in range(len(states)):
        emission_matrix[i,:] = map(np.longfloat,lines[i+9+len(states)].split('\t')[1:len(symbols)+1])
    return emission,symbols,states,transition_matrix,emission_matrix


text = 'xyxzzxyxyy\n--------\nx y z\n--------\nA B\n--------\n	A	B\nA	0.641	0.359\nB	0.729	0.271\n--------\n	x	y	z\nA	0.117	0.691	0.192	\nB	0.097	0.42	0.483\n'
path = 'AAABBAAAAA'
assert  hmm_decoding(*parse_emission_symbols_states_transition_matrix_emission_matrix(text)) == path


text = 'zxxxxyzzxyxyxyzxzzxzzzyzzxxxzxxyyyzxyxzyxyxyzyyyyzzyyyyzzxzxzyzzzzyxzxxxyxxxxyyzyyzyyyxzzzzyzxyzzyyy\n--------\nx y z\n--------\nA B\n--------\n	A	B\nA	0.634	0.366	\nB	0.387	0.613	\n--------\n	x	y	z\nA	0.532	0.226	0.241\nB	0.457	0.192	0.351\n'
path = 'AAAAAAAAAAAAAABBBBBBBBBBBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABBBBBBBBBBBAAAAAAAAAAAAAAAAAAAAABBBBBBBBBBAAA'
assert  hmm_decoding(*parse_emission_symbols_states_transition_matrix_emission_matrix(text)) == path
    

def hmm_emission_likelihood_prob(emission,symbols,states,transition_matrix,emission_matrix):
    """
    CODE CHALLENGE: Solve the Outcome Likelihood Problem.
    Input: A string x, followed by the alphabet from which x was constructed,
    followed by the states States, transition matrix Transition, and emission matrix
    Emission of an HMM (Σ, States, Transition, Emission).
    Output: The probability Pr(x) that the HMM emits x.
    """
    assert transition_matrix.shape == (len(states),len(states))
    assert emission_matrix.shape == (len(states),len(symbols))
    forward = np.empty(shape = (len(states),len(emission)), dtype = np.longfloat)
    
    forward[:,0] = 1./len(states)*emission_matrix[:,symbols[emission[0]]]
    for i in range(1,len(emission)):
        for k in range(len(states)):
            si = symbols[emission[i]]
            pforward = np.array(map(lambda l:forward[l,i-1]*transition_matrix[l,k], range(len(states))))
            forward[k,i] = pforward.sum()*emission_matrix[k,si]
#    print forward
    return forward[:,-1].sum()
    

np.set_printoptions(precision =32)
   
text = 'xzyyzzyzyy\n--------\nx y z\n--------\nA B\n--------\n	A	B\nA	0.303	0.697 \nB	0.831	0.169\n--------\n	x	y	z\nA	0.533	0.065	0.402\nB	0.342	0.334	0.324\n'
assert np.abs(hmm_emission_likelihood_prob(*parse_emission_symbols_states_transition_matrix_emission_matrix(text))-1.1005510319694847e-06) < 0.000000000000001
#
#s = hmm_emission_likelihood_prob(*parse_emission_symbols_states_transition_matrix_emission_matrix(text))
#print("challenge : %.10e" % s)

text = 'zxxxzyyxyzyxyyxzzxzyyxzzxyxxzyzzyzyzzyxxyzxxzyxxzxxyzzzzzzzxyzyxzzyxzzyzxyyyyyxzzzyzxxyyyzxyyxyzyyxz\n--------\nx y z\n--------\nA B\n--------\n	A	B\nA	0.994	0.006	\nB	0.563	0.437	\n--------\n	x	y	z\nA	0.55	0.276	0.173	\nB	0.311	0.368	0.321\n'
assert np.abs(hmm_emission_likelihood_prob(*parse_emission_symbols_states_transition_matrix_emission_matrix(text))-3.3318672795e-55) < 0.000000000000001
#
#s = hmm_emission_likelihood_prob(*parse_emission_symbols_states_transition_matrix_emission_matrix(text))
#print("extradataset : %.10e" % s)

############################################################
fpath = 'C:/Users/ngaude/Downloads/'
#fpath = '/home/ngaude/Downloads/'
#fpath = 'C:/Users/Utilisateur/Downloads/'
############################################################

#fname = fpath + 'dataset_11594_2.txt'
#with open(fname, "r") as f:
#    print hmm_path_prob(*parse_path_states_transition_matrix(f.read()))
    
#fname = fpath + 'dataset_11594_4.txt'
#with open(fname, "r") as f:
#    print hmm_emission_prob(*parse_emission_symbols_path_states_emission_matrix(f.read()))

#fname = fpath + 'dataset_11594_6.txt'
#with open(fname, "r") as f:
#    print hmm_decoding(*parse_emission_symbols_states_transition_matrix_emission_matrix(f.read()))

fname = fpath + 'dataset_11594_8.txt'
with open(fname, "r") as f:
    text = f.read()
    s = hmm_emission_likelihood_prob(*parse_emission_symbols_states_transition_matrix_emission_matrix(text))
    print(">> %.10e" % s)
