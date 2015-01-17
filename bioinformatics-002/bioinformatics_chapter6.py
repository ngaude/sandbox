# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 13:42:39 2015

@author: ngaude
"""

def greedy_sorting(p):
    '''
    CODE CHALLENGE: Implement GREEDYSORTING.
    Input: A permutation P.
    Output: The sequence of permutations corresponding to 
    applying GREEDYSORTING to P, ending with
    the identity permutation.
    '''
    s = []
    def reversal(i,k):
        prefix = p[:i-1]
        suffix = p[k+1:]
        root = p[i-1:k+1]
        return prefix + map(lambda x: -x, root[::-1]) + suffix
    for i in range(1,len(p)+1):
        try:
            k = p.index(-i)
            p = reversal(i,k)
            s.append(p)
        except ValueError:
            k = p.index(i)
            if (k != i-1):
                p = reversal(i,k)
                s.append(p)
                p = p[:]
                p[i-1] = -p[i-1]
                s.append(p)
    return s

def permutation_list_to_str(p):
    def str_val(i):
        if (i>0):
            return '+'+str(i)
        else:
            return str(i)
    return '(' + ' '.join(map(str_val,p)) + ')'

def permutation_str_to_list(str_p):
    p = map(int,str_p.strip()[1:-1].split(' '))
    return p
        
def format_sequence(s):
    fs = []
    for p in s:
        str_p = permutation_list_to_str(p)
        fs.append(str_p)
    return fs
    
assert len(greedy_sorting([-3,4,1,5,-2])) == 7

def number_of_breakpoints(p):
        '''
        CODE CHALLENGE: Solve the Number of Breakpoints Problem.
        
        Number of Breakpoints Problem: Find the number of breakpoints in a permutation.
        Input: A permutation.
        Output: The number of breakpoints in this permutation.        
        '''
        adj = 0
        p = [0,] + p + [len(p)+1]
        for i in range(0,len(p)-1):
            if (p[i+1]==p[i]+1):
                adj += 1
        return len(p) - 1 - adj

assert number_of_breakpoints([3, 4, 5, -12, -8, -7, -6, 1, 2, 10, 9, -11, 13, 14]) == 8


fname = 'C:/Users/ngaude/Downloads/number_of_breaks.txt'
with open(fname, "r") as f:
    p = f.read()
p = permutation_str_to_list(p)
print number_of_breakpoints(p)

fname = 'C:/Users/ngaude/Downloads/dataset_287_4.txt'
with open(fname, "r") as f:
    p = f.read()
p = permutation_str_to_list(p)
print number_of_breakpoints(p)



#p = '(100 '+' '.join(map(str,range(1,100)))+')'
#p = permutation_str_to_list(p)
#s = greedy_sorting(p)
#fs = format_sequence(s)
#print len(fs)

#fname = 'C:/Users/ngaude/Downloads/dataset_286_3.txt'
#with open(fname, "r") as f:
#    p = f.read()
#p = p = permutation_str_to_list(p)
#s = greedy_sorting(p)
#fs = format_sequence(s)
#with open(fname+'.out', "w") as f:
#    for p in fs:
#        f.write(p+'\n')

