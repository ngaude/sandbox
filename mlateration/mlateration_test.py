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


#
#ml = mlaterationsolver(Xi = [[46,59], [39,66], [32,17]], Di = [ 20,21,24])
#
#ml.solve()
#
#
#
#ml = mlaterationsolver(Xi = [[460,590], [390,660], [320,170]], Di = [ 200,210,240])
#ml.solve()
#
#
#
#
#
#
#
#ml = mlaterationsolver(Xi = [ [0,0], [1,0], [1,1], [0,1] ], Di = [2, 1, math.sqrt(2), math.sqrt(5)])
#X= ml.solve()
#print 'solution shall be  (2,0)'
#print 'solution', X
#print '--------'
#
#ml = mlaterationsolver(Xi = [ [0,0], [1,0], [0,1] ], Di = [math.sqrt(2), 1, 1])
#X= ml.solve()
#print 'solution shall be  (1,1)'
#print 'solution', X
#print '--------'
#



#ml = mlaterationsolver(Xi = [ [0,0], [1,0], [1,1], [0,1] ], Di = [2, 1, math.sqrt(2), math.sqrt(5)])
#X= ml.solve()
#print 'solution shall be  (2,0)'
#print 'solution', X
#print '--------'


ml = mlaterationsolver(Xi = [ [0,0], [1,0], [2,0] ], Di = [2,4,8])
X= ml.solve()
print 'no solution, optimum around (-4.5, 0)'
print 'solution', X
print '--------'