# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 13:42:39 2015

@author: ngaude
"""

_end = '_end_'

def make_trie(*words):
    """
    CODE CHALLENGE: Solve the Trie Construction Problem.
    Input: A collection of strings Patterns.
    Output: The adjacency list corresponding to Trie(Patterns), in the following format. If
    Trie(Patterns) has n nodes, first label the root with 0 and then label the remaining nodes with
    the integers 1 through n - 1 in any order you like. Each edge of the adjacency list of
    Trie(Patterns) will be encoded by a triple: the first two members of the triple must be the
    integers labeling the initial and terminal nodes of the edge, respectively; the third member
    of the triple must be the symbol labeling the edge.
    """
    root = dict()
    for word in words:
        current_dict = root
        for letter in word:
            current_dict = current_dict.setdefault(letter, {})
        current_dict = current_dict.setdefault(_end, _end)
    return root

def trie_in(trie,word):
    """
    return True/False)
    whether word is contained in trie
    """
    candidates = trie.keys()
    if _end in candidates:
        return True
    if not word:
        return False
    letter = word[0]
    suffix = word[1:]
    if letter in candidates:
        return trie_in(trie[letter],suffix)
    return False

def trie_tostr(root):
    s = []
    def dump_leaf(curr,parent_id):
        current_id = parent_id + 1
        for key, value in curr.iteritems():
            if (value == _end) or (value == _end):
                continue
            s.append(str(parent_id)+'->'+str(current_id)+':'+key)
            current_id = dump_leaf(value,current_id)
        return current_id
    dump_leaf(root,0)
    return '\n'.join(s)
    
def trie_matching(text,*words):
    """
    CODE CHALLENGE: Implement TRIEMATCHING to solve the Multiple Pattern Matching Problem.
    Input: A string Text and a collection of strings Patterns.
    Output: All starting positions in Text where a string from Patterns appears as a substring.
    """
    t = make_trie(*words)
    l = [i for i in range(len(text)) if trie_in(t,text[i:])]
    return l
    
    
assert trie_matching('AATCGGGTTCAATCGGGGT','ATCG','GGGT') == [1,4,11,15]  
   
######################
fname = 'C:/Users/ngaude/Downloads/dataset_294_8.txt'
with open(fname, "r") as f:
    l = f.read().splitlines()
    text = l[0]
    words = l[1:]
l = trie_matching(text,*words)
print ' '.join(map(str,l))
  
#fname = 'C:/Users/ngaude/Downloads/dataset_294_4.txt'
#with open(fname, "r") as f:
#    words = f.read().splitlines()
#t = make_trie(*words)
#s = trie_tostr(t)
#with open(fname+'.out', "w") as f:
#    f.write(s)
  

