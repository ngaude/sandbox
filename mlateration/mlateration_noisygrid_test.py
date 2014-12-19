# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 21:04:43 2014

@author: ngaude
"""

from mlateration import mlaterationgraph


import math
import random
import matplotlib.pyplot as plt


random.seed(123456)

g = mlaterationgraph()
for i in range (40):
    while True:
        x = random.randint(1, 10)
        y = random.randint(1, 10)
        u = (x,y)
        if not u in g.pos:
            break
    g.add_position(u,[x,y])

for i in range(500):
    noise = 0.005
    while True:
        xy = [random.randint(1, 10) for i in range(4)]
        d = math.sqrt(math.pow(xy[0] - xy[2],2) + math.pow(xy[1] - xy[3],2))
        d = d * (1.0 + random.uniform(-noise,noise))
        u = (xy[0],xy[1])
        v = (xy[2],xy[3])
        if (d > 0) and u != v and (not u in g.edge or not v in zip(*g.edge.get(u,[]))[0]):
            break
    g.add_edge(u, v, d)

fixed = list(g.pos.itervalues())
g.step = 100
g.solve(complete=True)
solved = list(g.pos.itervalues())

plt.xlim(0,11)
plt.ylim(0,11)
plt.scatter(*zip(*solved),color='blue')
plt.scatter(*zip(*fixed),color='red')
plt.show()

