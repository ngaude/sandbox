# -*- coding: utf-8 -*-
"""
author : nicolas gaude
date : 2014-12-18
"""
from mlateration import mlaterationsolver,mlaterationgraph


import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def tsv(name):
    datadir = '/home/ngaude/workspace/data/'
    return datadir + name + '.tsv'


E = pd.read_csv(tsv('ngaude_cell_graph_sensing_edge'), sep = '\t').dropna()

V = pd.read_csv(tsv('ngaude_cell_graph_sensing_vertex'), sep = '\t').dropna().drop_duplicates('numr_cell')

#estimate the average speed accross the graph

EV = E.merge(V, how = 'inner', left_on = 'numr_cell', right_on = 'numr_cell').merge(V, how = 'inner', left_on = 'nnumr_cell', right_on = 'numr_cell', suffixes = ('','_n'))

def speed(e):
    d = math.sqrt(math.pow(e.lon - e.lon_n, 2) + math.pow(e.lat - e.lat_n, 2))
    return d/e.dt

median_speed = np.median(EV.apply(speed, axis = 1))

g = mlaterationgraph()

for i,v in V.iterrows():
    g.add_position(v.numr_cell, [v.lon,v.lat])
    if (i%1000 == 0):
        print 'load positions to graph : ',i,'/',len(V)

for i,e in E.iterrows():
    g.add_edge(e.numr_cell, e.nnumr_cell, e.dt*median_speed)
    if (i%1000 == 0):
        print 'load edges to graph : ',i,'/',len(E)


g.solve()


#score the given solution
for k,v in g.pos.iteritems():
    (xa,ya) = k
    (xb,yb) = v
    err = math.sqrt(math.pow(xa-xb,2) + math.pow(ya-yb,2))
    if (err>0.1):
        print 'error',k,'!=',v

