# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 13:42:39 2015

@author: ngaude
"""

_end = '$'

def suffix_trie(text):
    root = dict()
    for i in range(len(text)):
        word = text[i:]
        current_dict = root
        for letter in word:
            if letter == _end:
                value = i
            else:
                value = {}
            current_dict = current_dict.setdefault(letter, value)
#        current_dict = current_dict.setdefault(_end, i)
    return root

def suffix_tree(text):
    """
    CODE CHALLENGE: Solve the Suffix Tree Construction Problem.
    Input: A string Text.
    Output: The edge labels of SuffixTree(Text). You may return these strings in any order.
    """
    root = suffix_trie(text)
    
    def find_nonbranching_path(label,current_dict):
        while (type(current_dict) == dict) and (len(current_dict.keys())==1):
            letter = current_dict.keys()[0]
            label += letter
            if (letter == _end):
                return (label,current_dict[letter])
                break
            current_dict = current_dict[letter]
        return (label,current_dict)
    
    def factor_nonbranching_path(current_dict):
        if type(current_dict) != dict:
            return
        labels = current_dict.keys()
        for label in labels:
            l,d = find_nonbranching_path(label,current_dict[label])
            current_dict.pop(label)
            current_dict[l] = d
            factor_nonbranching_path(d)
        return

    factor_nonbranching_path(root)
    return root

def suffix_tree_edges(current_dict):
    s = []
    if type(current_dict) != dict:
        return s
    for label,child_dict in current_dict.iteritems():
        s.append(label)
        s += suffix_tree_edges(child_dict)
    return s
    
    
    
text = 'CCAATAATTTACCGAACGGCTGCACTGGAAGTTCTAGTCGGACAAAAGGTAGGGCCACATTGACCATAATCGTCGCTTTGTACAGTCAAGACAGAGTATTGCAGATCACTGGTCATGAAATCCCGGAAACTGCAAGACTCCTCTAGTGTTTATCATTGCCTGTACGCTAATCGAGCGCGATAACTTGTCACTCTCCGATGCTTTCAGGACAGACAATCCTCTTCGCCCGCTCGTATTCACCCCAACCAAAACCCTGACTATGCGGGGCACTCCTACAACGAGAGGGCGTGTAAATACTTTTCGACACAACTGCTCTTCTCCCTCAACGTGGGTTGATGACGTTAAGGCCGGCATGTGTTACTAACCAGACGGGAGAATATCTTAAGTCCTCAACCGTCGAGTTTCTGGAACTGCAAGATGTTAGCAAGGCTACAGGAACGCAACAGGTCCTAGGTATACTCTTAATTGAACTGACAAAGTTTAAGGGCGAGACGCTCCGTATGTCCTACGGTAGGACTTAGCCGTAGCCCATTAGTGTCTCTCCACGCGGTATTGACAAGGGACCAGGAAATCTGCCGCCGTGTACCATGACACGCCCAAACTGTCCCGTGGTGTTCAGTCAAAGACATAATGGCACGTATACAAATTTCAACCAATATGACTACAAGAGCCGCGGACACAAGGAAAAAAGCCTACAAAGCTGTAGTCGGACGATTCCGTTTAACGCTCGTCGGTGCCACCCCGAGAGCGCTGCCCCCTGTTTATGGCGGCACAAGGCAAGGGGTTGCATGTCTGCTTGCCGTCTTCGAAGTTCAACGGTAGATATCCTCTAAACACATCAGCTCCTGTCAACCTATTCACATCCTCTTATAATCCACCTATCTGCCTTCAGAGTCCCCCTGGTGCACCTACGCGATGCCGCTGTGAATTACGACCTCGGCGAATCGCAGGGAACTCGAGAATTTTTATCCGTGAGTTGTATGGTGTTGGTTTCCTCCAACGGTGTCCCGGAATCGTTTGCGCACACGGGGCCCTGGTATGCGGTATTAAACCCGCGAGTCTAAAACGCTTGACTCCTTTAGGTGACTTTCGTACATCACTTGCGGTCCAGAGTGCAAGGCATATCTCGATGTGCCGCAGCCTGCGCTCTAATAGCCATCTATTCGTTAGCCAGCTCGCTGAATCCTCTCTTTTTTGTCGATATATTAGAGAGGAAGTACGGTAAACTCGGCTCCGGGTGTGGGATAGGGTAAGTACGAACTGCGAAATTTATTTTGTTCGCACAGTGATTAGTCTAGAGATTAGCCATTATTTTGTCGTTGCGCCTACGCCGAAATATGATTCTCCGTTTTCCCGCAGCACGCTGACGTTAAAATTGCCAAAACAGATGACGGTAACCCGAAACCAAGGGCCCAGGAGATCCGCTTGCGACCATTACAA$'
#text = 'ATAAATG$'
aa = suffix_trie(text)
bb = suffix_tree(text)

ee = suffix_tree_edges(bb)

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
            if (value == _end):
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

fname = 'C:/Users/ngaude/Downloads/dataset_296_4.txt'
with open(fname, "r") as f:
    l = f.read().splitlines()
    text = l[0]
t = suffix_tree(text)
e = suffix_tree_edges(t)
with open(fname+'.out', "w") as f:
    f.write('\n'.join(e))

#fname = 'C:/Users/ngaude/Downloads/dataset_294_8.txt'
#with open(fname, "r") as f:
#    l = f.read().splitlines()
#    text = l[0]
#    words = l[1:]
#l = trie_matching(text,*words)
#print ' '.join(map(str,l))
  
#fname = 'C:/Users/ngaude/Downloads/dataset_294_4.txt'
#with open(fname, "r") as f:
#    words = f.read().splitlines()
#t = make_trie(*words)
#s = trie_tostr(t)
#with open(fname+'.out', "w") as f:
#    f.write(s)