# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 17:39:44 2014

@author: ngaude
"""

import numpy as np

w = [-1,1]
b = -2

xs= [(3,4), (5,4), (7,8)]

cl = {(3,4) : 1, (5,4) : -1, (7,8) : 1}


for x in xs:
    print 'slack ',x,"=",max(0, 1 - cl.get(x,0)*(np.dot(x,w)+b))