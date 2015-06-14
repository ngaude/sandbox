# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 10:04:39 2015

@author: ngaude
"""

import numpy as np

def hmm_profile(theta,alphabet,multalign):
    """
    CODE CHALLENGE: Solve the Profile HMM Problem.
    Input: A threshold θ, followed by an alphabet Σ, followed by a multiple alignment
    Alignment whose strings are formed from Σ.
    Output: The transition matrix followed by the emission matrix of HMM(Alignment, θ).
    Note: Your matrices can be either space-separated or tab-separated.
    """
    a = np.array(map(list,multalign))
    (n,m) = a.shape
    # compute column id where insertion fraction is greater than theta
    col_is_m = np.apply_along_axis(lambda r:np.sum(r == '-')<n*theta,0,a)
    col_to_state = np.cumsum(col_is_m)
    # compute the symbols translation table
    symbols = {s: i for (i,s) in enumerate(alphabet)}    
    
    # compute the name_state translation table
    max_state_id = max(col_to_state)
    name_state = {}
    name_state['S'] = 0
    name_state['I0'] = 1
    for i in range(1,max(col_to_state)+1):
        name_state['M'+str(i)] = 3*i - 1
        name_state['D'+str(i)] = 3*i
        name_state['I'+str(i)] = 3*i + 1
    name_state['E'] = 3*max_state_id + 2
    state_name = [None] * len(name_state)
    for k,v in name_state.iteritems():
        state_name[v] = k
    
    # emission
    emission = np.zeros((len(name_state),len(symbols)))
    for i in range(n):
        for j in range(m):
            insert_flag = not col_is_m[j]
            state_id = col_to_state[j]
            e = a[i,j]
            if e in symbols:
                emission[name_state[('I' if insert_flag else 'M')+str(state_id)],symbols[e]] += 1       


    # transition
    transition = np.zeros((len(name_state),len(name_state)))
    for i in range(n):
        prev_state_name = 'S'
        for j in range(m):
            state_id = col_to_state[j]
            insert_flag = not (col_is_m[j])
            delete_flag = not (a[i,j] in symbols)
            if insert_flag and delete_flag:
                # nop nop nop
                assert True == True 
            elif (not insert_flag) and (not delete_flag):
                curr_state_name ='M' + str(state_id)
                transition[name_state[prev_state_name],name_state[curr_state_name]] += 1
                prev_state_name = curr_state_name
            elif insert_flag and (not delete_flag):
                curr_state_name ='I' + str(state_id)
                transition[name_state[prev_state_name],name_state[curr_state_name]] += 1
                prev_state_name = curr_state_name
            elif (not insert_flag) and delete_flag:
                curr_state_name ='D' + str(state_id)
                transition[name_state[prev_state_name],name_state[curr_state_name]] += 1
                prev_state_name = curr_state_name
    i = name_state['M'+str(max_state_id)]
    transition[i,name_state['E']] = transition[:,i].sum()
    i = name_state['D'+str(max_state_id)]
    transition[i,name_state['E']] = transition[:,i].sum()
    i = name_state['I'+str(max_state_id)]
    transition[i,name_state['E']] = transition[:,i].sum()  
    # normalize matrices probability
    for i in name_state.values():
        csum = emission[i,:].sum()
        if (csum > 0):
            emission[i,:] /= csum
    emission = np.round(emission,3)
#    for i in name_state.values():
#        csum = transition[i,:].sum()
#        if (csum > 0):
#            transition[i,:] /= csum
#    transition = np.round(transition,3)
    return transition,emission,alphabet,state_name


def print_hmm_profile(transition,emission,alphabet,state_name):
    ret = ''
    #ret += '\t' + '\t'.join(state_name) + '\n'
    ret += '\t' + '\t'.join(state_name) + '\t\n'
    for i,s in enumerate(state_name) :
         l = s + '\t'
         l += '\t'.join([format("%.3g" % transition[i,j]) if transition[i,j] != 1 else '1.0' for j in range(len(state_name))])
         ret += l+ '\n'
    ret += '--------\n'
    ret += '\t' + '\t'.join(alphabet)
    for i,s in enumerate(state_name) :
         l = '\n'+ s + '\t'
         l += '\t'.join([format("%.3g" % emission[i,j]) if emission[i,j] != 1 else '1.0' for j in range(len(alphabet))])
         ret += l
    return ret


############################################################
fpath = 'C:/Users/ngaude/Downloads/'
#fpath = '/home/ngaude/Downloads/'
#fpath = 'C:/Users/Utilisateur/Downloads/'
############################################################

#theta = 0.289
#alphabet = ['A','B','C','D','E']
#multalign = ['EBA','E-D','EB-','EED','EBD','EBE','E-D','E-D']
#(t,e,a,s) = hmm_profile(theta,alphabet,multalign)
#print_hmm_profile(t,e,a,s)

theta = 0.252
alphabet = ['A','B','C','D','E']
multalign = ['DCDABACED','DCCA--CA-','DCDAB-CA-','BCDA---A-','BC-ABE-AE']
(t,e,a,s) = hmm_profile(theta,alphabet,multalign)
ret = print_hmm_profile(t,e,a,s)

fname = fpath + 'profileHMM.txt'
with open(fname, "r") as f:
    text = f.read()
    

t = text.split('\n')
r = ret.split('\n')

assert len(t) == len(r)

#assert ret == text


#theta = 0.275
#alphabet = ['A','B','C','D','E']
#multalign = ['ADECDCCA-','ADECDEEAA','CDBCDAEA-','BDEE-E-AD','ADE-DEA-D','ADECDEAA-','-DB-EEDAD']
#(t,e,a,s) = hmm_profile(theta,alphabet,multalign)
#print_hmm_profile(t,e,a,s)

#theta = 0.236
#alphabet = ['A','B','C','D','E']
#multalign = ['ADCCECD--','CECAD-ACA','CDCADCA-A','ADCADCACA','BDCACCACA','BDC-DEAC-','-BCADCACA']
#(t,e,a,s) = hmm_profile(theta,alphabet,multalign)
#print_hmm_profile(t,e,a,s)


