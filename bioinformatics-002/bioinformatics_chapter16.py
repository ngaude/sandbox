# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 10:04:39 2015

@author: ngaude
"""

import numpy as np

############################################################
fpath = 'C:/Users/ngaude/Downloads/'
#fpath = '/home/ngaude/Downloads/'
#fpath = 'C:/Users/Utilisateur/Downloads/'
############################################################

def hmm_profile(theta,alphabet,multalign,pseudocount = 0):
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
            if (insert_flag) and (delete_flag):
                if j == m-1:
                    transition[name_state[prev_state_name],name_state['E']] += 1
            else:
                if (not insert_flag) and (not delete_flag):
                    curr_state_name ='M' + str(state_id)
                elif insert_flag and (not delete_flag):
                    curr_state_name ='I' + str(state_id)
                elif (not insert_flag) and delete_flag:
                    curr_state_name ='D' + str(state_id)
                else:
                    print insert_flag,delete_flag
                    assert True == False #cannot happen anyway....
                transition[name_state[prev_state_name],name_state[curr_state_name]] += 1
                prev_state_name = curr_state_name
                # last column specific case
                if j == m-1:            
                    transition[name_state[prev_state_name],name_state['E']] += 1
 
    # normalize matrices probability
    for i in name_state.values():
        csum = emission[i,:].sum()
        if (csum > 0):
            emission[i,:] /= csum
    for i in name_state.values():
        csum = transition[i,:].sum()
        if (csum > 0):
            transition[i,:] /= csum
    
    # adding pseudocount if needed
    if pseudocount > 0:
        # emission :
        # add pseudocount only for 'I' and 'M' states
        m,n = emission.shape
        for i in range(1,m-1):
            if (i%3):
                emission[i,:] += pseudocount
        for i in name_state.values():
            csum = emission[i,:].sum()
            if (csum > 0):
                emission[i,:] /= csum
        # emission :
        # add transition only for allowed transition states
        m,n = transition.shape
        for i in range(0,m-1):
            a = min((i+1)/3*3+1,n)
            b = min((i+1)/3*3+4,n)
            transition[i,a:b] += pseudocount
        for i in name_state.values():
            csum = transition[i,:].sum()
            if (csum > 0):
                transition[i,:] /= csum
    
    return transition,emission,alphabet,state_name


def print_hmm_profile(transition,emission,alphabet,state_name):
    __emission = np.round(emission,3)
    __transition = np.round(transition,3)
    ret = ''
    #ret += '\t' + '\t'.join(state_name) + '\n'
    ret += '\t' + '\t'.join(state_name) + '\n'
    for i,s in enumerate(state_name) :
         l = s + '\t'
         l += '\t'.join([format("%.3g" % __transition[i,j]) if __transition[i,j] != 1 else '1.0' for j in range(len(state_name))])
         ret += l+ '\n'
    ret += '--------\n'
    ret += '\t' + '\t'.join(alphabet)
    for i,s in enumerate(state_name) :
         l = '\n'+ s + '\t'
         l += '\t'.join([format("%.3g" % __emission[i,j]) if __emission[i,j] != 1 else '1.0' for j in range(len(alphabet))])
         ret += l
    return ret


def hmm_profile_decoding(x,symbols,states,transition,emission):
    """
    CODE CHALLENGE: Implement the Viterbi algorithm solving the Decoding Problem.
    Input: A string x, followed by the alphabet from which x was constructed,
    followed by the states States, transition matrix Transition, and emission matrix
    Emission of an HMM (Σ, States, Transition, Emission).
    Output: A path that maximizes the (unconditional) probability Pr(x, π) over all possible paths π.
    """
    assert transition.shape == (len(states),len(states))
    assert emission.shape == (len(states),len(symbols))
    rsymbols = {k: i for (i,k) in enumerate(symbols)}
    rstates = {k: i for (i,k) in enumerate(states)}
        
    n = len(x)+1
    m = len(states)/3
    dscore = np.empty(shape = (m,n), dtype = float)
    mscore = np.empty(shape = (m,n), dtype = float)
    iscore = np.empty(shape = (m,n), dtype = float)

    
    dbackt = np.empty(shape = (m,n), dtype = tuple)
    mbackt = np.empty(shape = (m,n), dtype = tuple)
    ibackt = np.empty(shape = (m,n), dtype = tuple)

    # fill up values with None
    dbackt.fill(None)
    mbackt.fill(None)
    ibackt.fill(None)

    # fill up values with 666
    dscore.fill(666)
    mscore.fill(666)
    iscore.fill(666)

    # initialize non-reachable values with 777
    dscore[0,:] = 777
    mscore[:,0] = 777
    mscore[0,:] = 777
    iscore[1:,0] = 777

    def recurrence_M(l,k):
        assert l>1 and k>1
        d_prev_id = rstates['D'+str(l-1)]
        m_prev_id = rstates['M'+str(l-1)]
        i_prev_id = rstates['I'+str(l-1)]
        m_curr_id = rstates['M'+str(l)]
        k_id = rsymbols[x[k-1]]
        score_M = mscore[l-1,k-1] + np.log(emission[m_curr_id,k_id]*transition[m_prev_id,m_curr_id])
        score_D = dscore[l-1,k-1] + np.log(emission[m_curr_id,k_id]*transition[d_prev_id,m_curr_id])
        score_I = iscore[l-1,k-1] + np.log(emission[m_curr_id,k_id]*transition[i_prev_id,m_curr_id])
        score_max = max(score_M,score_D,score_I)
        if score_M == score_max:
            return (score_M,('M',l-1,k-1))
        if score_D == score_max:
            return (score_D,('D',l-1,k-1))
        if score_I == score_max:
            return (score_I,('I',l-1,k-1))

    def recurrence_D(l,k):
        d_prev_id = rstates['D'+str(l-1)]
        m_prev_id = rstates['M'+str(l-1)]
        i_prev_id = rstates['I'+str(l-1)]
        d_curr_id = rstates['D'+str(l)]
        score_M = mscore[l-1,k] + np.log(1*transition[m_prev_id,d_curr_id])
        score_D = dscore[l-1,k] + np.log(1*transition[d_prev_id,d_curr_id])
        score_I = iscore[l-1,k] + np.log(1*transition[i_prev_id,d_curr_id])
        score_max = max(score_M,score_D,score_I)
        if score_M == score_max:
            return (score_M,('M',l-1,k))
        if score_D == score_max:
            return (score_D,('D',l-1,k))
        if score_I == score_max:
            return (score_I,('I',l-1,k))

    def recurrence_I(l,k):
        d_curr_id = rstates['D'+str(l)]
        m_curr_id = rstates['M'+str(l)]
        i_curr_id = rstates['I'+str(l)]
        k_id = rsymbols[x[k-1]]
        score_M = mscore[l,k-1] + np.log(emission[i_curr_id,k_id]*transition[m_curr_id,i_curr_id])
        score_D = dscore[l,k-1] + np.log(emission[i_curr_id,k_id]*transition[d_curr_id,i_curr_id])
        score_I = iscore[l,k-1] + np.log(emission[i_curr_id,k_id]*transition[i_curr_id,i_curr_id])
        score_max = max(score_M,score_D,score_I)
        if score_M == score_max:
            return (score_M,('M',l,k-1))
        if score_D == score_max:
            return (score_D,('D',l,k-1))
        if score_I == score_max:
            return (score_I,('I',l,k-1))    

#    print '----------'
#    print 'initialize non-reachable values with 777'
#    print 'mscore='
#    print mscore
#    print
#    print 'dscore='
#    print dscore
#    print
#    print 'iscore='
#    print iscore
#    print '----------'
    
    
    # initialize viterbi graph top row I0
    i0_id = rstates['I0']
    s_id = rstates['S']
    k_id = rsymbols[x[0]]
    iscore[0,1] = np.log(emission[i0_id,k_id]*transition[s_id,i0_id])
    ibackt[0,1] = None
    for k in range(2,n):
        i0_id = rstates['I0']
        k_id = rsymbols[x[k-1]]
        iscore[0,k] = iscore[0,k-1] + np.log(emission[i0_id,k_id]*transition[i0_id,i0_id])
        ibackt[0,k] = ('I',0,k-1)
    
#    print '----------'
#    print 'initialize viterbi graph top row I0'
#    print 'iscore='
#    print iscore
#    print '----------'

    # initialize viterbi graph top row M1
    m1_id = rstates['M1']
    s_id = rstates['S']
    k_id = rsymbols[x[0]]
    mscore[1,1] = np.log(emission[m1_id,k_id]*transition[s_id,m1_id])
    mbackt[1,1] = None  
    for k in range(2,n):
        m1_id = rstates['M1']
        i0_id = rstates['I0']
        k_id = rsymbols[x[k-1]]
        mscore[1,k] = iscore[0,k-1] + np.log(emission[m1_id,k_id]*transition[i0_id,m1_id])
        mbackt[1,k] = ('I',0,k-1)

#    print '----------'
#    print 'initialize viterbi graph top row M1'
#    print 'mscore='
#    print mscore
#    print '----------'
        
    # initialize viterbi graph top row D1
    d1_id = rstates['D1']
    s_id = rstates['S']
    dscore[1,0] = np.log(1*transition[s_id,d1_id])
    dbackt[1,0] = None    
    for k in range(1,n):
        i0_id = rstates['I0']
        d1_id = rstates['D1']
        dscore[1,k] = iscore[0,k] + np.log(1*transition[i0_id,d1_id])
        dbackt[1,k] = ('I',0,k)

#    print '----------'
#    print 'initialize viterbi graph top row D1'
#    print 'dscore='
#    print dscore
#    print '----------'

    # initialize viterbi graph first left column D0,D1,....
    dscore[1,0] = 0
    dbackt[1,0] = None
    for l in range(2,m):
        d_prev_id = rstates['D'+str(l-1)]
        d_curr_id = rstates['D'+str(l)]
        dscore[l,0] = dscore[l-1,0] + np.log(1*transition[d_prev_id,d_curr_id])
        dbackt[l,0] = ('D',l-1,0)
    
#    print '----------'
#    print 'initialize viterbi first left column D0,D1,....'
#    print 'dscore='
#    print dscore
#    print '----------'
    

    # initialize viterbi graph second left column I1,I2,....
    for l in range(1,m):
        d_curr_id = rstates['D'+str(l)]
        i_curr_id = rstates['I'+str(l)]
        k_id = rsymbols[x[0]]
        iscore[l,1] = dscore[l,0] + np.log(emission[i_curr_id,k_id]*transition[d_curr_id,i_curr_id])
        ibackt[l,1] = ('D',l,0)
        
#    print '----------'
#    print 'initialize viterbi second left column I1,I2,....'
#    print 'iscore='
#    print iscore
#    print '----------'
    
    # initialize viterbi graph second left column M2,M3,....
    for l in range(2,m):
        d_prev_id = rstates['D'+str(l-1)]
        m_curr_id = rstates['M'+str(l)]
        k_id = rsymbols[x[0]]
        mscore[l,1] = dscore[l-1,0] + np.log(emission[m_curr_id,k_id]*transition[d_prev_id,m_curr_id])
        mbackt[l,1] = ('D',l-1,0)

#    print '----------'
#    print 'initialize viterbi second left column M2,M3,....'
#    print 'mscore='
#    print mscore
#    print '----------'

    # recurrence on I1 row
    for k in range(2,n):
        (iscore[1,k],ibackt[1,k]) = recurrence_I(1,k)

#    print '----------'
#    print 'recurrence on I1 row'
#    print 'iscore='
#    print iscore
#    print '----------'
    
    #recurrence on second second left column D2,D3,....
    for l in range(2,m):
        (dscore[l,1],dbackt[l,1]) = recurrence_D(l,1)
        
#    print '----------'
#    print 'recurrence on I1 row'
#    print 'iscore='
#    print iscore
#    print '----------'
    
    # recurrence from the 2,2 corner...
    for l in range(2,m):
            for k in range(2,n):
               (mscore[l,k],mbackt[l,k]) = recurrence_M(l,k)
               (dscore[l,k],dbackt[l,k]) = recurrence_D(l,k)
               (iscore[l,k],ibackt[l,k]) = recurrence_I(l,k)
    
    
  
#    print '----------'
#    print 'recurrence from the 2,2 corner...'
#    print 'mscore='
#    print mscore
#    print
#    print 'dscore='
#    print dscore
#    print
#    print 'iscore='
#    print iscore
#    print '----------'  

    # backtracking max score from backt pointers

    mm_id = rstates['M'+str(m-1)]
    dm_id = rstates['D'+str(m-1)]
    im_id = rstates['I'+str(m-1)]
    e_id = rstates['E']
    score_M = mscore[m-1,n-1] + np.log(transition[mm_id,e_id])
    score_D = dscore[m-1,n-1] + np.log(transition[dm_id,e_id])
    score_I = iscore[m-1,n-1] + np.log(transition[im_id,e_id])
    score_max = max(score_M,score_D,score_I)

    hidden_path = []
    if score_M == score_max:
        hidden_path.append('M'+str(m-1))
        (state,l,k) = mbackt[m-1,n-1]
    elif score_D == score_max:
        hidden_path.append('D'+str(m-1))
        (state,l,k) = dbackt[m-1,n-1]
    elif score_I == score_max:
        hidden_path.append('I'+str(m-1))
        (state,l,k) = ibackt[m-1,n-1]
    
    while True:
        if state == 'M':
            hidden_path.append('M'+str(l))
            backt = mbackt
        if state == 'D':
            hidden_path.append('D'+str(l))
            backt = dbackt
        if state == 'I':
            hidden_path.append('I'+str(l))
            backt = ibackt
        if backt[l,k] == None:
            break
        (state,l,k) = backt[l,k]
           
    return ' '.join(hidden_path[::-1])

def hmm_profile_hidden_path(x,theta,pseudocount,alphabet,multalign):
    """
    CODE CHALLENGE: Solve the Sequence Alignment with Profile HMM Problem.
    Input: A string x followed by a threshold θ and a pseudocount σ, followed by an
    alphabet Σ, followed by a multiple alignment Alignment whose strings are formed 
    from Σ. 
    Output: An optimal hidden path emitting x in HMM(Alignment, θ, σ).
    """
    (t,e,a,s) = hmm_profile(theta,alphabet,multalign,pseudocount = pseudocount)
    ret = hmm_profile_decoding(x,a,s,t,e)
    return ret
    


x = 'AEFDFDC'
theta = 0.4 
pseudocount = 0.01
alphabet = ['A','B','C','D','E','F']
multalign = ['ACDEFACADF','AFDA---CCF','A--EFD-FDC','ACAEF--A-C','ADDEFAAADF']
assert 'M1 D2 D3 M4 M5 I5 M6 M7 M8' == hmm_profile_hidden_path(x,theta,pseudocount,alphabet,multalign)

x = 'EEBEABDCEEABCCCEEBDEDCADEDACCDCBBEECDBDACABDADCBE'

theta = 0.359
pseudocount = 0.01
alphabet = ['A','B','C','D','E']
multalign = ['EEBBA--C-DBAA-AECD--BDB---CC-DDCBBCEDE-EBB-DAEE-C','EEEEABBCEABBCDEE-DAEBDBAEDC-BDBCB--C-B-BCA-DAEECA','--CEB-ACCDEACEEEEDBEED-ADBCCDAC--BC--BDBCAEDAEECC','A--AABDCE-A-CD-ECD-EBBA-EDC-DACCBBCCD-D-BA-DAAEBC','EECEAB--EDDACCE-CD-E--B-EDCCD-CCBBCCD-DBBA--AE-CA','E-CDA-DCECAAECB-CDCEB-B-BDCCD---B--CD-DBCDBDAEB-C','EBCEAEDC-DABC--A-DCEDDBAED-CD-CCBBCCEBDB--BEA-EEC','AC-E-BDCEDAADDEECDEEB-CAEDC-DD-CBBCCD-DBCABDAEECC','EECBABDCEDEAEC-DCDC-BDBDEDA-D-AD-A-EABEB--BDA-ECC']

#(t,e,a,s) = hmm_profile(theta,alphabet,multalign,pseudocount = pseudocount)
#print print_hmm_profile(t,e,a,s)

sol = 'M1 M2 M3 M4 M5 M6 M7 M8 M9 D10 M11 M12 I12 I12 M13 M14 M15 M16 M17 M18 D19 M20 M21 M22 M23 I23 M24 M25 M26 I26 I26 M27 D28 M29 M30 M31 I31 I31 D32 M33 M34 I34 M35 M36 M37 M38 I38 M39 M40 M41 M42 M43 M44'
ret = hmm_profile_hidden_path(x,theta,pseudocount,alphabet,multalign)
assert sol == ret

def hmm_parameter_estimation(x,alphabet,path,states):
    """
    CODE CHALLENGE: Solve the HMM Parameter Estimation Problem.
    Input: A string x of symbols emitted from an HMM, followed by the HMM's alphabet Σ,
    followed by a path π, followed by the collection of states of the HMM.
    Output: A transition matrix Transition followed by an emission matrix Emission that maximize
    Pr(x, π) over all possible transition and emission matrices.
    """
    assert len(x)==len(path)
    m = len(alphabet)
    n = len(states)
    p = len(path)
    transition = np.zeros((n,n))
    emission = np.zeros((n,m))
    rsymbols = {k: i for (i,k) in enumerate(alphabet)}
    rstates = {k: i for (i,k) in enumerate(states)}
        
    for i in range(p):
        sy_id = rsymbols[x[i]]
        st_id = rstates[path[i]]
        emission[st_id,sy_id] += 1.
    
    for i in range(1,p):
        curr_st_id = rstates[path[i]]
        prev_st_id = rstates[path[i-1]]
        transition[prev_st_id,curr_st_id] += 1

    def matrix_norm(mtx):
        a,b = mtx.shape
        for i in range(a):
            csum = mtx[i,:].sum()
            if csum == 0:
                mtx[i,:] = 1./b
            else:
                mtx[i,:] /=  1.*csum
        return

    matrix_norm(transition)
    matrix_norm(emission)

    return transition,emission

    
np.set_printoptions(precision=10)
x = 'yzzzyxzxxx'
alphabet = ['x','y','z']
path = 'BBABABABAB'
states = ['A','B','C']
(t,e) = hmm_parameter_estimation(x,alphabet,path,states)
t = np.round(t,3)
e = np.round(e,3)
ret = print_hmm_profile(t,e,alphabet,states)


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

def parse_emission_symbols_states_transition_matrix_emission_matrix(lines):
    emission = lines[0]
    symbols = lines[2].split(' ')
    states = lines[4].split(' ')
    transition_matrix = np.zeros((len(states),len(states)), dtype = np.longfloat)
    for i in range(len(states)):
        transition_matrix[i,:] = map(np.longfloat,lines[i+7].split('\t')[1:len(states)+1])
    emission_matrix = np.zeros((len(states),len(symbols)), dtype = np.longfloat)
    for i in range(len(states)):
        emission_matrix[i,:] = map(np.longfloat,lines[i+9+len(states)].split('\t')[1:len(symbols)+1])
    return emission,symbols,states,transition_matrix,emission_matrix

def hmm_viterbi_learning(it,x,symbols,states,transition,emission):
    """
    CODE CHALLENGE: Implement Viterbi learning for estimating the parameters of an HMM.
    Input: A number of iterations j, followed by a string x of symbols emitted by an HMM,
    followed by the HMM's alphabet Σ, followed by the HMM's states, followed by initial transition
    and emission matrices for the HMM.
    Output: Emission and transition matrices resulting from applying Viterbi learning for j iterations.
    """
    dsymbols = {s: i for (i,s) in enumerate(symbols)}
    dstates = {s: i for (i,s) in enumerate(states)}
    for j in range(it):
        path = hmm_decoding(x,dsymbols,dstates,transition,emission)
        (transition,emission) = hmm_parameter_estimation(x,symbols,path,states)
        print print_hmm_profile(transition,emission,symbols,states)
        print '**************************************'
    print 
    return transition,emission

#fname = fpath + 'ViterbiLearning.txt'
#with open(fname, "r") as f:
#    text = f.read()
#    lines = text.split('\n')
#    it = int(lines[0])
#    x,symbols,states,t,e = parse_emission_symbols_states_transition_matrix_emission_matrix(lines[2:])
#    t,e = hmm_viterbi_learning(it,x,symbols,states,t,e)
#    ret = print_hmm_profile(t,e,symbols,states)
#with open(fname+'.out', "w") as f:
#    text = f.write(ret)    


############################################################
############################################################
############################################################
############################################################



#theta = 0.252
#alphabet = ['A','B','C','D','E']
#multalign = ['DCDABACED','DCCA--CA-','DCDAB-CA-','BCDA---A-','BC-ABE-AE']
#(t,e,a,s) = hmm_profile(theta,alphabet,multalign)
#ret = print_hmm_profile(t,e,a,s)
#fname = fpath + 'profileHMM.txt'
#with open(fname, "r") as f:
#    text = f.read()
#t = text.split('\n')
#r = ret.split('\n')
#assert len(t) == len(r)
#assert t == r
#assert ret == text
#
#theta = 0.399
#pseudocount = 0.01
#alphabet = ['A','B','C','D','E']
#multalign = ['ED-BCBDAC','-D-ABBDAC','ED--EBD-C','-C-BCB-D-','AD-BC-CA-','-DDB-BA-C']
#(t,e,a,s) = hmm_profile(theta,alphabet,multalign,pseudocount = pseudocount)
#ret = print_hmm_profile(t,e,a,s)
#fname = fpath + 'profileHMMPseudocounts.txt'
#with open(fname, "r") as f:
#    text = f.read()
#t = text.split('\n')
#r = ret.split('\n')
#assert len(t) == len(r)
#assert t == r
#assert ret == text
#



#fname = fpath + 'dataset_11632_2.txt'
#with open(fname, "r") as f:
#    text = f.read()
#    lines = text.split('\n')
#    theta = float(lines[0])
#    alphabet = lines[2].split('\t')
#    multalign = [l for l in lines[4:] if len(l)==len(lines[4])]
#    print text
#    print theta,alphabet,multalign
#(t,e,a,s) = hmm_profile(theta,alphabet,multalign)
#sol = print_hmm_profile(t,e,a,s)
#with open(fname+'.out', "w") as f:
#    text = f.write(sol)
#
#fname = fpath + 'dataset_11632_4.txt'
#with open(fname, "r") as f:
#    text = f.read()
#    lines = text.split('\n')
#    theta = float(lines[0].split(' ')[0])
#    pseudocount = float(lines[0].split(' ')[1])
#    alphabet = lines[2].split(' ')
#    multalign = [l for l in lines[4:] if len(l)==len(lines[4])]
#(t,e,a,s) = hmm_profile(theta,alphabet,multalign,pseudocount = pseudocount)
#sol = print_hmm_profile(t,e,a,s)
#with open(fname+'.out', "w") as f:
#    text = f.write(sol)

#x = 'BBDDEBDABECADABECECEDEBBAACBDACAECBCCADABECEECCED'
#theta = 0.231
#pseudocount = 0.01
#alphabet = ['A','B','C','D','E']
#multalign = ['-B-BEBDABE-CEABBCCA-D-ABA--BB-CCECBCCC-DDBEEACAED','-B--DB--ADD-DBB--CAEDE-BAA-EEB-CDDBCC--D-AEE-CAED','BDD-E-DAB-AE--BACCAA-E-DA-EBCA-CD-BCCEDD-AEEDCAED','-BB-EDDAB-CBDADACDBE--AAAAE-D-CCECBCECBDECEBABAED','BB-B-BDABECEDABD-CA-DE-DEA-BDCCCCCBCC-DAD-EEA-BED']
#ret = hmm_profile_hidden_path(x,theta,pseudocount,alphabet,multalign)

#fname = fpath + 'HMMParameterEstimation.txt'
#fname = fpath + 'dataset_11632_8.txt'
#with open(fname, "r") as f:
#    text = f.read()
#    lines = text.split('\n')
#    x = lines[0]
#    alphabet = lines[2].split(' ')
#    path = lines[4]
#    states = lines[6].split(' ')
#(t,e) = hmm_parameter_estimation(x,alphabet,path,states)
#ret = print_hmm_profile(t,e,alphabet,states)
#with open(fname+'.out', "w") as f:
#    text = f.write(ret)

fname = fpath + 'dataset_11632_10.txt'
with open(fname, "r") as f:
    text = f.read()
    lines = text.split('\n')
    it = int(lines[0])
    x,symbols,states,t,e = parse_emission_symbols_states_transition_matrix_emission_matrix(lines[2:])
    t,e = hmm_viterbi_learning(it,x,symbols,states,t,e)
    ret = print_hmm_profile(t,e,symbols,states)
with open(fname+'.out', "w") as f:
    text = f.write(ret)