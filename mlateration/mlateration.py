# -*- coding: utf-8 -*-
"""
m-lateration solver & graph iterator
simple implementation of m-lateration position estimation 
given distance and reference point.
graph solver : given a distance graph and some know points,
try to solve all points position.
author : nicolas gaude
date : 2014-21-17
"""

import math
import numpy as np

####################################
###
#
#

class mlaterationsolver:

    epsilon = np.finfo(float).eps    
    
    def __init__(self, Xi, Di, lr = 0.01, step = 100):
        '''
        create a m-lateration solver for 
        - m 'Xi' 2d-points [xi,yi]
        - m 'Di' distances di
        perfect solution is a 2d-point X [x,y] 
        where X is for all i<m at exactely Di distance from point Xi
        if no perfect solution can be found, approximation minimizing
        m 'Di to Xi' constaints with gradient descent and learning rate r
        '''
#        assert len(Di) > 2
        assert len(Xi) == len(Di)
        self.m = len(Di)
        self.Xi = np.array(Xi).reshape(self.m, 2)
        self.Di = np.array(Di).reshape(self.m)
        self.Di2 = np.square(self.Di)
        self.lr = lr
        self.step = step
        return

    def __B(self, X):
        '''
        compute a m vector = [ ..., sqrt ( (x-xi)^2 + (y-yi)^2 ), ....]
        '''
        B = np.sqrt(np.square(self.__XXi(X)).sum(axis = 0))
        B += + mlaterationsolver.epsilon
        return B
    
    def __A(self, X):
        '''
        compute a m vector = [ ..., (x-xi)^2 + (y-yi)^2 - di^2, ....]
        '''
        A = np.square(self.__XXi(X)).sum(axis = 0) - self.Di2
        
        return A


    '''
    grad2 & cost2 are minimizing sum(i)  (d(X,Xi) - Di)^2
    '''    

    def grad(self, X):
        '''
        compute gradient of m-lateration at X for distance (sqrt(x^2)^2 poly)
        grad is a vector = __B * __XXi
        '''
        
        # compute dXXi = Di*XXi/B
        # add epsilon to B to avoid the zero-division error
        dXXi = np.multiply(self.__XXi(X),np.divide(self.Di,self.__B(X)))
        return (self.__XXi(X) - dXXi).sum(axis=1)

    def cost(self, X):    
        return np.sqrt(np.square(self.__B(X) - self.Di).sum())
        

    '''
    grad2 & cost2 are minimizing sum(i)  (d(X,Xi)^2 - Di^2)^2
    '''    
    
    def grad2(self, X):
        '''
        compute gradient of m-lateration at X for squared-distance (x^4 poly)
        grad is a vector = __XXi * __A
        '''
        return np.dot(self.__XXi(X), self.__A(X))

    def cost2(self, X):
        '''
        compute cost of m-lateration for X
        X vector = [x,y] is a solution of m-lateration is cost = 0
        cost = sum i<m  (xi-x)^2+(y-yi)^2 - di^2
        '''
        return math.pow(np.square(self.__A(X)).sum(),0.25)

    # debugging purpose
    # shall be put private after     
    def XXi(self,X):
        return self.__XXi(X)
    def A(self,X):
        return self.__A(X)
    def B(self,X):
        return self.__B(X)

    
    def __XXi(self, X):
        '''
        compute a m*2  matrix of vector = [ ..., [X-Xi], ...]
        where Xi is a vector = [xi,yi]
        '''
        assert len(X) == 2
        return np.subtract(X,self.Xi).transpose()
    
        
    
    def iterate(self, X):
        '''
        each iteration returns:
        - an update X position vector
        - the dX move vector
        '''
        assert len(X) == 2
        grad = self.grad(X)
        
#        norm  = np.linalg.norm(grad)
#        if (norm < 1):
#            dX = - self.lr * grad
#        else:
#            dX = - self.lr * grad/norm
        dX = - self.lr * grad
        X = np.array(X) + dX
        return (X,dX)
    
    def solve(self):
        '''
        return a 'local-optimal' solution of m-lateration
        '''
        # initialize the X candidate a the centroid of Xi's
        X = self.Xi.sum(axis=0)/(1.0*len(self.Xi))
        dX = self.grad(X)
        cost = self.cost(X)
        best_cost = (cost, X, self.lr)
    
        i = 0
        while (i < self.step):
            i += 1
            prev_dX = dX
            (X,dX) = self.iterate(X)
            cost = self.cost(X)
            if cost > best_cost[0]:
                # cost back_tracking to the best cost position
                # and decrease its learning rate
                (cost,X,self.lr) = best_cost
                self.lr /= 4
                best_cost = (cost, X, self.lr)
            else:
                self.lr *= 2
                best_cost = (cost, X, self.lr)
                    
#            else:
#                # gradient back-tracking
#                if np.dot(prev_dX, dX) > 0:
#                    # everything is fine : cost decrease & forward gradient
#                    best_cost = (cost, X, self.lr)
#                    self.lr *= 2
#                else:
#                    # not so good : cost decrease but rewind gadient
#                    self.lr /= 2
#                    X = X-dX
            
            
#            print i,'iteration X=',X,'dX=',dX,'cost=',self.cost(X),'lr=',self.lr
        return X


class mlaterationgraph:
    ''' 
    a lateration graph is an undirected distance graph
    some position vertex are known
    to solve others position,
    iterate to find position for missing vertex position using m-lateration
    '''
    def __init__(self, dfunc = None):
        # dict of known vertex position
        self.pos = {}
        # set of vertex
        self.vertex = set()
        # adjacency list of set of edge : 
        # key = vertex initial
        # value = ( ..., (vertex final, distance), ...)
        self.edge = {}
        self.step = 100
        self.stage = 0
        self.limit = 3
        # dfunc : is a function parameter. used to compute a distance equivalent to parameter 't' at position 'pos'
        self.dfunc = dfunc
        return
        
    def add_edge(self, u, v, d):
        '''
        add an undirected distance d edge between vertex u and v
        '''
        assert u != v
        self.vertex.add(u)
        self.vertex.add(v)
        self.edge.setdefault(u,set()).add((v,d))
        self.edge.setdefault(v,set()).add((u,d))
    
    def add_position(self, u, X):
        '''
        add a new 2d-point position information for vertex u
        '''
        assert len(X) == 2
        assert u
        assert not u in self.pos
        self.vertex.add(u)
        self.pos[u] = X
        
    
    def unsolved(self):
        return self.vertex.difference(self.pos.keys())
    
    def solvable(self):
        '''
        return the list of 
        solvable position with their associated m-lateration points
        number of known points m shall be at least 3 per solvable position
        '''
        solvable = []
        unsolved = self.vertex.difference(set(self.pos.keys()))
        
        for u in unsolved:
            l = filter(lambda v: v[0] in self.pos, self.edge[u])
            if len(l) >= self.limit:
                (V, Ti) = zip(*l)
                Xi = [ self.pos[v] for v in V ]
                solvable.append((u, Xi, Ti))
        return solvable
    
    def iterate(self):
        solved = []
        solvable = self.solvable()
        for i,(u, Xi, Ti) in enumerate(solvable):

            if (self.dfunc is None):
                # if dfunc is None, then Di == Ti e.g constant speed of  along the graph 1
                Di  = Ti
            else:
                # else, then use dfunc to convert Ti to Di given position of Xi's centroid
                Di = [self.dfunc(x,t) for x,t in zip(Xi,Ti)]

            ml = mlaterationsolver(Xi,Di,step = self.step)
            X = ml.solve()
#            print u,'=',X,' solved ',i,'/',len(solvable), ' lr = ',ml.lr,'cost = ',ml.cost(X)
            self.add_position(u, X)
            solved.append(u)
            if (i%1000 == 0):
                print 'round',self.stage,': solved=',len(self.pos),'/',len(self.vertex),'solving=',i,'/',len(solvable)
        return solved
    
    def solve(self, complete = False):
        self.stage =0
        while self.stage < 1000:
            print 'round', self.stage,': unsolved positions =', len(self.unsolved())
            solved = self.iterate()
            if len(self.unsolved())==0:
                break
            self.stage += 1
            if not solved:
                unsolvable = self.unsolved()
                print 'unsolvable (<'+str(self.limit)+') position = ', len(unsolvable)
                if complete and self.limit > 1:
                    self.limit-=1
                else:
                    break
        return
