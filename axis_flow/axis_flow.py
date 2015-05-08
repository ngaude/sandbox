# -*- coding: utf-8 -*-
"""
Created on Fri May 08 13:04:09 2015

@author: ngaude

"""

import numpy as np
from scipy import optimize
from scipy import signal

# setup :
# we consider an oriented railway axis made of  :
# * n+1, railway stations : station(i) ordered along the axis from start to end
# * n, railway segments : segment(i) defined as rail section from station i to i+1
# * n*(n+1)/2, railway trunks : trunk(i,j ) defined as rail sections from station i to j (i<j)

# problem :
# given flow information for any station of axis, 
# compute the segment/trunk load

# input : 
# flow information matrix is defined an ndarray of dimension 3: i*t*f
# * 1st dim : i, is the index of a station : i in [0,n]
# * 2nd dim : t, is the timestamp of the considered time interval [t,t+dt]
#       dt = 1 minute. period = 1 day e.g : t in [0,1199]
#       day starts at 4:30:01 and finished the day after at 4:30:00
#       e.g : t = 125 corresponds to [06:35:01,06:36:01] time interval
# * 3rd dim : f, is the index of the type of flow considered : 
#       f = 0 : flow_in : people flowing in to station i between [t,t+dt]
#       f = 1 : flow_through : people flowing through station i between [t,t+dt]
#       f = 2 : flow_out : people flowing out from station i between [t,t+dt]

# output :
# * segment_load(i,t) : returns the load on the railway between station i and i+1 at timestamp t
# * trunk_load(i,j,t) : returns the load on the railway between station i and j at timestamp t

def flow_normed(flow):
    # returns a normalized a flow matrix : 
    # the returns flow matrix will fit the n given constraints :
    # * for any segment i, on the period, the following total flows shall be equal :
    #       people moving on segment i from station i : flow_through + flow_in
    #       people moving on segment i to station i+1 : flow_through+ flow_out
    # * the correcting factors applied shall be as close as possible from 1
    
    # total amount of people moving on segment i from station i on period 
    ttrin = flow[:,:,0].sum(axis=1) + flow[:,:,1].sum(axis=1)
    
    # total amount of people moving on segment i to station i+1 on period
    ttrout = flow[:,:,1].sum(axis=1) + flow[:,:,2].sum(axis=1)
    
    def flow_pacing(a):
        n = flow.shape[0]-1
        # regularization is the importance of the 'unitary factor constraint'
        # only needed if we encounter severe factors distortion ....
        regularization = 0.
        cost = 0
        for i in range(n):
            c = (a[i]*ttrin[i] - a[i+1]*ttrout[i+1])
            c = c*c
            cost += c
        for i in range(n+1):
            c = (a[i]-1)
            c = c*c
            cost += regularization*c
        return cost            
    alpha = np.ones(flow.shape[0])
    alpha = optimize.fmin(flow_pacing, alpha)
    nflow = np.copy(flow)
    print alpha
    for i in range(flow.shape[0]):
        nflow[i,:,:] *= alpha[i]
    return nflow

def segment_traversal_time(flow,i):
    # given flow and segement i, returns the most probable traversal_time 
    # of segment i in t units along the whole period.
    #
    # todo : check if traversal time evolves significantly at peek hours
    # refinement : traversal time could be computed on a per t range basis...

    trin = flow[i,:,0] + flow[i,:,1]
    trout = flow[i+1,:,1] + flow[i+1,:,2]

    # because of traversal time nature,
    # trini and trouti are idealy dephased of traversal time
    # e.g. it exits tt s.a : trin(t-tt) = trout(t) for any t
    # thus, tt shall maximise the correlation of trout to trin

    tt = np.argmax(signal.correlate(trout,trin)) - (len(trin)-1)
    return tt
    
def traversal_time(flow):
    tts = map(lambda i:segment_traversal_time(flow,i), range(flow.shape[0]-1))
    return np.array(tts)



######################################################
# testing
######################################################

def print_period_segment_load(flow):
    ttrin = flow[:,:,0].sum(axis=1) + flow[:,:,1].sum(axis=1)
    ttrout = flow[:,:,1].sum(axis=1) + flow[:,:,2].sum(axis=1)
    print ttrin
    print ttrout
    
def add_train(flow,people,g1,t1,g2,t2):
    # add people in a train that :
    # * flow_in at gare g1 at t1
    # * flow_out to gare g2 at t2
    step = np.linspace(t1,t2,1+g2-g1).astype(int)
    print len(step),step
    flow[g1,step[0],0] = people
    for i,t in zip(range(g1+1,g2),step[1:-1]):
        flow[i,t,1] = people
    flow[g2,step[-1],2] = people

# toy flow matrix for testing purpose    
test_flow = np.zeros((8,1200,3))
# add a 100-people in train from gare 0 at 6AM to gare 7 at 9:30AM
add_train(test_flow,100,0,90,7,300)
# add a 200-people in train from gare 0 at 10AM to gare 7 at 13:30AM
add_train(test_flow,200,0,330,7,630)
# add +10% flow detection perturbance at gare 3
test_flow[3,:,:] *= 1.1
# add -20% flow detection perturbance at gare 5
test_flow[5,:,:] *= 0.8

print test_flow.shape

# that moves station 10mn along axis
print_period_segment_load(test_flow)
test_flow = flow_normed(test_flow)
print_period_segment_load(test_flow)

tt = traversal_time(test_flow)
print tt