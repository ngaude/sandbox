# -*- coding: utf-8 -*-
"""
author : nicolas gaude
date : 2014-12-18
"""
from mlateration import mlaterationsolver,mlaterationgraph


import math
import random
import numpy as np

np.seterr(all = 'raise')



ml = mlaterationsolver(Xi = [[46,59], [39,66], [32,17]], Di = [ 20,21,24])

ml.solve()



ml = mlaterationsolver(Xi = [[460,590], [390,660], [320,170]], Di = [ 200,210,240])
ml.solve()







ml = mlaterationsolver(Xi = [ [0,0], [1,0], [1,1], [0,1] ], Di = [2, 1, math.sqrt(2), math.sqrt(5)])
X= ml.solve()
print 'solution shall be  (2,0)'
print 'solution', X
print '--------'

ml = mlaterationsolver(Xi = [ [0,0], [1,0], [0,1] ], Di = [math.sqrt(2), 1, 1])
X= ml.solve()
print 'solution shall be  (1,1)'
print 'solution', X
print '--------'




ml = mlaterationsolver(Xi = [ [0,0], [1,0], [1,1], [0,1] ], Di = [2, 1, math.sqrt(2), math.sqrt(5)])
X= ml.solve()
print 'solution shall be  (2,0)'
print 'solution', X
print '--------'


ml = mlaterationsolver(Xi = [ [0,0], [1,0], [2,0] ], Di = [2,4,8])
X= ml.solve()
print 'no solution, optimum around (-4.5, 0)'
print 'solution', X
print '--------'

# simple test square-grid 10x10 positions
# 20 known positions
# 500 known edges with some noise
# try to establish 80 missing positions...

g = mlaterationgraph()
for i in range (20):
    while True:
        x = random.randint(0, 10)
        y = random.randint(0, 10)
        u = (x,y)
        if not u in g.pos:
            break
    g.add_position(u,[x,y])

for i in range(500):
    while True:
        xy = [random.randint(0, 10) for i in range(4)]
        d = math.sqrt(math.pow(xy[0] - xy[2],2) + math.pow(xy[1] - xy[3],2))
        d = d * (1.0 + random.uniform(-0.1,-0.1))
        u = (xy[0],xy[1])
        v = (xy[2],xy[3])
        if (d > 0) and u != v and (not u in g.edge or not v in zip(*g.edge.get(u,[]))[0]):
            break
    g.add_edge(u, v, d)

g.solve()
#score the given solution
for k,v in g.pos.iteritems():
    (xa,ya) = k
    (xb,yb) = v
    err = math.sqrt(math.pow(xa-xb,2) + math.pow(ya-yb,2))
    if (err>0.1):
        print 'error',k,'!=',v

