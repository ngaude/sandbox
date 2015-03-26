# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:04:28 2015

@author: ngaude
"""
import cmath
import numpy as np

def cost(abc):
    abc1 = sum(map(lambda a:a,abc))
    abc2 = sum(map(lambda a:a*a,abc))
    abc3 = sum(map(lambda a:a*a*a,abc))
    return abs(abc1-3)+abs(abc2-5)+abs(abc3-7)

optimal = (1000,None)

oa = 1.62000550
oa = 1.62000591
oa = 1.62000590
oa = 1.62000590
oa = 1.62000590486
oa = 1.62000590486



ob = 0.391435
ob = 0.39143579
ob = 0.39143577
ob = 0.39143578
ob = 0.3914357753
ob = 0.391435775292



eps= 0.00000000005

for a in np.linspace(oa-eps,oa+eps,1000):
    for b in np.linspace(ob-eps,ob+eps,1000):
        c = 3-2*a
        abc = np.array([a+b*1j, a-b*1j, c], dtype=complex)
        res = cost(abc)
        if optimal[0] > res:
            optimal = (res,abc)
            print optimal[0]
            print 'oa=',optimal[1][0].real
            print 'ob=',optimal[1][0].imag

solution = optimal[1]
reponse = sum(map(lambda a:a*a*a*a,solution))

#a = 1.62
#for b in np.linspace(0.39142,0.39144,1000000):
#    c = 3-2*a
#    abc = np.array([a+b*1j, a-b*1j, c], dtype=complex)
#    res = cost(abc)
#    if optimal[0] > res:
#        optimal = (res,abc)
#        print optimal

#        
#a = 1.5j
#ab = np.array([1+a, 1-a], dtype=complex)
#print sum(map(lambda a:a*a*a,ab))

#solution = np.array([ 1.61961962-0.39039039j,  1.61961962+0.39039039j, -0.23923924+0.j])
#
#np.array(1.666)