# -*- coding: utf-8 -*-
"""
author : nicolas gaude
date : 2014-12-18
"""
from mlateration import mlaterationgraph


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import spatial


def tsv(name):
    datadir = '/home/ngaude/workspace/data/'
    return datadir + name + '.tsv'


E = pd.read_csv(tsv('ngaude_cell_graph_sensing_edge'), sep = '\t').dropna()

V = pd.read_csv(tsv('ngaude_cell_graph_sensing_vertex'), sep = '\t').dropna().drop_duplicates('numr_cell')

#estimate the average speed accross the graph

EV = E.merge(V, how = 'inner', left_on = 'numr_cell', right_on = 'numr_cell').merge(V, how = 'inner', left_on = 'nnumr_cell', right_on = 'numr_cell', suffixes = ('','_n'))




def speed(e):
    min_speed_kmh = 10
    max_speed_kmh = 100
    min_speed_deg = min_speed_kmh/3.6*360/40000000
    max_speed_deg = max_speed_kmh/3.6*360/40000000
    d = math.sqrt(math.pow(e.lon - e.lon_n, 2) + math.pow(e.lat - e.lat_n, 2))
    v = d/(e.dt+1)
    if (v < min_speed_deg):
        v = min_speed_deg
    elif (v > max_speed_deg):
        v = max_speed_deg
    return v

def mid(e):
    return ((e.lon + e.lon_n)/2, (e.lat + e.lat_n)/2)

def dfunc(pos,t):
    '''
    in a in-progress graph dfunc is used to dynamically
    compute a distance equivalent to parameter 't' at position 'pos'
    estimated given k nearest sampled points
    '''
    [dist,idx] = posvtree.query(pos, k = 123)
    v = [posv['v'][i] for i in idx]
    v = np.median(v)
    return v*t

posv = { 'pos' : list(EV.apply(mid, axis = 1)), 'v' : list(EV.apply(speed, axis = 1)) }
posvtree = spatial.cKDTree(posv['pos'])

g = mlaterationgraph(dfunc = dfunc)

for i,v in V.iterrows():
    g.add_position(v.numr_cell, [v.lon,v.lat])
    if (i%1000 == 0):
        print 'load positions to graph : ',i,'/',len(V)

for i,e in E.iterrows():
    g.add_edge(e.numr_cell, e.nnumr_cell, e.dt)
    if (i%1000 == 0):
        print 'load edges to graph : ',i,'/',len(E)

g.step = 100
g.solve(complete=True)

with open(tsv('ngaude_cell_graph_computing_vertex'), 'a') as f:
    f.write('numr_cell\tlon\tlat\n')
    for k,v in g.pos.iteritems():
        f.write(k + '\t' + str(v[0]) + '\t' + str(v[1]) + '\n')
