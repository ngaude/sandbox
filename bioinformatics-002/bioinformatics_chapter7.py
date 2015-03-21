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
        print '===>',word
        current_dict = root
        for letter in word:
            current_dict = current_dict.setdefault(letter, {})
        current_dict = current_dict.setdefault(_end, _end)
    return root

def dump_trie(root):
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
    
def make_trie_l(*words):
    def insert_word(trie,word):
        if not word:
            return
        # given a root subtree insert letter and return new root
        letters = trie[0]
        branches = trie[1]
        letter = word[0]
        if letter in letters:
            # path already in trie
            i = letters.index(letter)
            ntrie = branches[i]
        else:
            # add a new sub-path to trie
            letters.append(letter)
            ntrie = ([],[])
            branches.append(ntrie)
        insert_word(ntrie,word[1:])
        return
            
    root = ([],[])
    
    for word in words:
        insert_word(root,word)
    return root
  
fname = 'C:/Users/ngaude/Downloads/dataset_294_4.txt'
#fname = 'C:/Users/ngaude/Downloads/TrieConstruction.txt'
with open(fname, "r") as f:
    words = f.read().splitlines()
t = make_trie(*words)
s = dump_trie(t)
with open(fname+'.out', "w") as f:
    f.write(s)
