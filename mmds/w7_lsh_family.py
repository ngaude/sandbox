# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 01:07:24 2014

@author: Utilisateur
"""

ts = ('abcef', 'acdeg', 'bcdefg', 'adfg', 'bcdfgh', 'bceg', 
'cdfg', 'abcd')

J = 0.79999

dprefix = {}

def prefix_length(s):
    return int(len(s)*(1-J))+1

def indexing(s):    
    l = prefix_length(s)
    def char_indexing(c):
        cset = dprefix.setdefault(c, set())
        cset.add(s)
    map(char_indexing, s[0:l])

def compare_set(s):
    vs = dprefix.values()
    ss = set()
    for cs in vs:
        if s in cs:
            ss.update(cs)
    ss.remove(s)
    return (s,len(ss))

print zip(ts,map(prefix_length,ts))
print '----------'
map(indexing,ts)
print map(compare_set,ts)
#print dprefix


