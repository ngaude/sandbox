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
# flow information matrix is defined as an ndarray of dimension 3: i*t*f
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
# * axis_load() : returns the load on all segments of axis at any timestamp t
#       the load axis matrix as an ndarray of dimension 2: i*t
#       * 1st dim : i, is the index of a segment : i in [0,n-1]
#       * 2nd dim : t, is the timestamp of the considered absolute time t
#                   t are 1 minute spaced. period = 1 day e.g : t in [0,1199]
#                   day starts at 4:30:00 and finished the day after at 4:30:00
#                   e.g : t = 125 corresponds to 06:35:00

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
    alpha = optimize.fmin(flow_pacing, alpha, maxfun = 1000000,maxiter = 1000000)
    nflow = np.copy(flow)
    print 'average correction per station',zip(range(len(alpha)),alpha)
    for i in range(flow.shape[0]):
        nflow[i,:,:] *= alpha[i]
    return nflow

def segment_traversal_time(flow,i):
    # given flow and segment i, returns the most probable traversal_time 
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

def segment_load(flow,dt=15):
    trin = flow[:,:,0] + flow[:,:,1]
    trout = flow[:,:,1] + flow[:,:,2]
    assert trin.shape == trout.shape
    (G,T) = trin.shape
    seg = np.zeros(shape=(G-1,T/dt))
    for g in range(G-1):
        for t in range(0,T,dt):
            seg[g,t/dt] = trin[g,t:t+dt].sum()
    return seg

def section_load(flow,dt=15):
    trin = flow[:,:,0] + flow[:,:,1]
    trout = flow[:,:,1] + flow[:,:,2]
    assert trin.shape == trout.shape
    (G,T) = trin.shape
    trc = np.zeros(shape=(G-1,T/dt))
    ctrin = trin.cumsum(axis=1)
    ctrout = trout.cumsum(axis=1)
    for i,g in enumerate(range(G-1)):
        for t in range(0,T,dt):
            if (i==0):
                trc[g,t/dt] = ctrin[g:g+i+1,t].sum()-ctrout[g+1:g+i+2,t].sum()
    return trc

def section_load_with_traversal(flow,dt=15,p=3):
    # NOTE: accumulation of noisy flow data along the day 
    # is giving more and more error as long as t move forward.
    # better idea is to integrate accumulation on a realistic time buffer only.
    # thus we do consider that the traversal time is a good indicator of train
    # propagation along the axis.
    # ...
    trin = flow[:,:,0] + flow[:,:,1]
    trout = flow[:,:,1] + flow[:,:,2]
    assert trin.shape == trout.shape
    (G,T) = trin.shape
    trc = np.zeros(shape=(G-1,T/dt))
    tts = traversal_time(flow)
    for i,g in enumerate(range(G-1)):
        for t in range(0,T,dt):
            ti = max(0,t-tts[i]-p*dt)
            to = max(0,t-p*dt)
            trc[g,t/dt] = trin[g:g+i+1,ti:t].sum()-trout[g+1:g+i+2,to:t].sum()
    return trc

######################################################
# main func
######################################################

def axis_load(flow):
    flow = flow_normed(flow)
    tt = traversal_time(test_flow)
    n = flow.shape[0]-1
    T = flow.shape[1]
    axis = np.zeros((n,T))
    for i in range(n):
        for t in range(T):
            axis[i,t] = max(segment_load(flow,i,t,tt=tt),0)
    return axis

######################################################
# load sncf station flow data
######################################################

def epoch_sec(t):
    # string datetime in a 'YYYY-mm-dd HH:MM:SS' format
    # return elapsed seconds from epoch time
    dt = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
    s = time.mktime(dt.timetuple())
    return s

def str_date(s):
    # s elapsed seconds from epoch time
    # return string datetime in a 'YYYY-mm-dd HH:MM:SS' format
    dt = datetime.datetime.fromtimestamp(s)
    sdt = dt.strftime('%Y-%m-%d %H:%M:%S')
    return sdt 

def daily_hour(d):
    t0 = epoch_sec(d[:10]+ ' 04:30:31')
    t = epoch_sec(d)
    return (((t-t0)+4*3600+30*60+1)% (24*3600))/3600.0

import pandas as pd
import datetime
import time
import matplotlib.pyplot as plt


fpath = '/home/ngaude/workspace/data/'
fname = fpath + 'sncf_flux_ligne_flow_agg_by_mn.tsv'

df = pd.read_csv(fname,sep="\t")

ligne = df[(df.gare_ligne == 'villeneuve-st-georges=>creil') & (df.horaire >= '2014-05-19 04:30') & (df.horaire < '2014-05-20 04:30') & (df.direction == 'villeneuve-st-georges=>creil')]

# ligne = df[(df.gare_ligne == 'villeneuve-st-georges=>creil') & (df.horaire >= '2014-05-19 04:30') & (df.horaire < '2014-05-20 04:30') & (df.direction == 'creil=>villeneuve-st-georges')]


#ligne = df[(df.gare_ligne == 'lyon=>dijon') & (df.horaire >= '2014-05-19 04:30') & (df.horaire < '2014-05-20 04:30') & (df.direction == 'lyon=>dijon')]

# ligne = df[(df.gare_ligne == 'lyon=>dijon') & (df.jour == '2014-05-19') & (df.direction == 'lyon=>dijon')]
# ligne = df[(df.gare_ligne == 'lyon=>dijon') & (df.direction == 'lyon=>dijon')]

flow = np.zeros((21,1440,3))
tstart = epoch_sec('2014-05-19 04:30:00')
for i,r in ligne.iterrows():
    tcurr = epoch_sec(r.horaire+':00')
    mn = int(tcurr-tstart)/60
    flow[r.gare_index-1,mn,0]=r.montant
    flow[r.gare_index-1,mn,1]=r.transit_par
    flow[r.gare_index-1,mn,2]=r.descendant

flow = flow_normed(flow)

seg = segment_load(flow,dt=15)
plt.figure('segment')
for i in range(len(seg)):
    splt = plt.subplot(4,5,i+1)
    splt.set_title(str(i)+'->'+str(i+1))
    plt.plot(seg[i,:])

plt.show()

#trc = section_load(flow,dt=15)
trc = section_load_with_traversal(flow,dt=15)
plt.figure('section_load')
for i in range(len(trc)):
    splt = plt.subplot(4,5,i+1)
    splt.set_title('0->'+str(i+1))
    plt.plot(trc[i,:])

plt.show()




for i in range(0,1,1):
    trin = flow[i,:,0]+flow[i,:,1]
    trout = flow[i+1,:,1]+flow[i+1,:,2]    
    trinp = peaky(trin,10)
    troutp = peaky(trout,10)
    fp = trin-trout
    plt.plot(fp,label=i,alpha=0.7)

plt.legend()
plt.show()

print sum(f)
print sum(fp)

plt.plot(f)
plt.plot(fo)
plt.show()



def peaky(f,o=3):
    fpeak = signal.argrelmax(f,order = o)[0]
    peak_mean = f[fpeak].mean()
    # consider that no train is less than half of average train
    limit = peak_mean/2.
    # collect volume around the peak
    ff = np.array([ f[i-o:i+o+1].sum() if (i in fpeak) and (x > limit) else 0 for i,x in enumerate(f)])
    scale = f.sum()/ff.sum()
    return ff*scale

trin = flow[:,:,0] + flow[:,:,1]
trout = flow[:,:,1] + flow[:,:,2]

peaky_flow(trin)

ctrin = trin.cumsum(axis=1)
ctrout = trout.cumsum(axis=1)

if ligne.direction.max() == ligne.gare_ligne.min():
    # when index has the same direction than ligne
    seg = ctrin[:-1] - ctrout[1:]
else:
    # when index has the opposite direction than ligne
    seg = ctrin[1:] - ctrout[:-1]


for i in range(20):
    plt.plot(seg[i].transpose(),label = i)
    plt.legend()
    plt.show(block=False)



######################################################
# testing
######################################################
    
def add_train(flow,people,g1,t1,g2,t2):
    # add people in a train that :
 

   # * flow_in at gare g1 at t1
    # * flow_out to gare g2 at t2
    step = np.linspace(t1,t2,1+g2-g1).astype(int)
    flow[g1,step[0],0] = people
    for i,t in zip(range(g1+1,g2),step[1:-1]):
        flow[i,t,1] = people
    flow[g2,step[-1],2] = people

# toy flow matrix for testing purpose    
# get inspired by voyages-scnf.com for realistic traversal time
test_flow = np.zeros((8,600,3))

# add a 1000-people in train from gare 0 at 08:02 to gare 7 at 9:33
add_train(test_flow,1000,0,212,7,333)

# add a 300-people in train from gare 0 at 08:53 to gare 7 at 10:34
add_train(test_flow,300,0,95,7,330)

# add a 200-people in train from gare 0 at 10:23 to gare 7 at 11:58
add_train(test_flow,200,0,383,7,458)

# add a 1200-people in train from gare 0 at 11:23 to gare 7 at 12:58
add_train(test_flow,1200,0,421,7,518)

# add a 700-people in train from gare 0 at 11:58 to gare 7 at 13:34
add_train(test_flow,700,0,458,7,554)

# add +10% flow detection perturbance at gare 3
test_flow[3,:,:] *= 1.1
# add -20% flow detection perturbance at gare 5
test_flow[5,:,:] *= 0.8

# check flow conservation consistency :
tt = traversal_time(flow_normed(test_flow))
print 'average station traversal time :',tt

# that's it :
axis = axis_load(test_flow)

# some drawing...
import matplotlib.pyplot as plt

plt.title('axis load')

for i in [0,2,4,6]:
    plt.plot(axis[i,:],label = 'segment '+str(i))
plt.legend(loc = 'upper left')
plt.show()


