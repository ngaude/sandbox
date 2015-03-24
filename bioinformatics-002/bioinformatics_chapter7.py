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

def suffix_tree_edges(text):
    def dfs(current_dict):
        if type(current_dict) != dict:
            return
        for label,child_dict in current_dict.iteritems():
            s.append(label)
            dfs(child_dict)
        return
    
    s = []
    root = suffix_tree(text)
    dfs(root)
    return s


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


def compare_string(a,b):
    minlen = min(len(a),len(b))
    count = 0
    for i in range(minlen):
        if (a[i]!=b[i]):
            break
        count += 1
    return count
            

def new_suffix_tree(text):
    root = dict()

    def insert_suffix(current_dict,j,i):
        if type(current_dict) != dict:
            print 'weird cannot insert suffix <',text[j:],'>'
            return
        keys = current_dict.keys()
        first_letters = [ text[k[0]] for k in keys]
        word = text[j:]
        if word[0] in first_letters:
            # at least one letter common path is found
            # check how long the common path is :
            # might be larger than word then add (i) inside key (a,b)
            # might be larger than label then recurisvely add it deeper
            # might be a split between key and word
            key = keys[first_letters.index(word[0])]
            label = text[key[0]:key[1]]
            count = compare_string(word,label)
            # print 'common_label',text[key[0]:key[0]+count],' with ',key
            
            if count == len(label):
                # recursive call here
                # print 'recursive insert (',i,text[i:],') remaining word=',text[j+count:],' in ',current_dict[key]
                insert_suffix(current_dict[key],j+count,i)
                return
            if count == len(word):
                # cannot happen, because of the $ sign
                print 'shall not happen'
            # remove the key from dict
            value = current_dict.pop(key)
            # create a child_dict from value and remaning string of label
            child_dict = {(key[0]+count,key[1]):value}
            # insert the new smaller common_label as a key
            current_dict[(key[0],key[0]+count)] = child_dict
            # insert the remaining part of word with (i)
            child_dict[(j+count,len(text))] = i
        else:
            # new path, insert completely word with (i)
            # print 'so is a new path in current_dict',current_dict,'with ',(j,len(text))
            current_dict[j,len(text)] = i
    for i in range(len(text)):
        # print 'insert >>',text[i:] 
        insert_suffix(root,i,i)
    return root

def new_suffix_tree_edges(text):
    t = new_suffix_tree(text)
    s = []
    def dfs_edges(current_dict):
        if type(current_dict) != dict:
            return
        for label,child_dict in current_dict.iteritems():
            s.append(text[label[0]:label[1]])
            dfs_edges(child_dict)
        return
    dfs_edges(t)
    return s        
   

def longest_repeat(text):
    """
    Longest Repeat Problem: Find the longest repeat in a string.
    Input: A string Text.
    Output: A longest repeat in Text, i.e., a longest substring of Text that appears in Text more than once.
    """
    def dfs(root,word):
        if (len(root.keys())>1) and (len(longests[-1]) < len(word)):
            # word is repreated and is larger than any longest's
            longests.append(word)

        for key, value in root.iteritems():
            if type(value) == dict:
                label = text[key[0]:key[1]]
                dfs(value,word+label)
        return
    longests = ['',]
    t = new_suffix_tree(text+'$')
    dfs(t,'')
    return longests[-1]



def tree_coloring(edge_lines,node_lines):
    """
    Tree Coloring Problem: Color the internal nodes of a tree given the colors of its leaves.
    Input: An adjacency list, followed by color labels for leaf nodes.
    0 -> {}
    1 -> {}
    2 -> 0,1
    3 -> {}
    4 -> {}
    5 -> 3,2
    6 -> {}
    7 -> 4,5,6
    -
    0: red
    1: red
    3: blue
    4: blue
    6: red    
    Output : Color labels for all nodes, in any order.
    0: red
    1: red
    2: red
    3: blue
    4: blue
    5: purple
    6: red
    7: purple    
    """
    tree_adjlist = [None]*(len(edge_lines))
    tree_color = {}
    for l in edge_lines:
        # parse edge parent node
        spl = l.strip().split(' -> ')
        parent_id = int(spl[0])
        if spl[1] != '{}':
            # (let leaves None as initialized)
            # parse edge children node
            children_ids = [int(c) for c in spl[1].split(',')]
            tree_adjlist[parent_id] = children_ids
    for l in node_lines:
        # parse node id
        spl = l.strip().split(': ')
        node_id = int(spl[0])
        if spl[1] == 'red':
            tree_color[node_id] = 1
        elif spl[1] == 'blue':
            tree_color[node_id] = 2
        else:
            print 'weird color (',spl[1],') here, shall not happen'
    def color_node(current_id):
        if current_id in tree_color.keys():
            return tree_color[current_id]
        current_list = tree_adjlist[current_id]
        assert current_list is not None
        color = 0
        for child_id in current_list:
            if child_id in tree_color.keys():
                color |= tree_color[child_id]
            else:
                color |= color_node(child_id)
        tree_color[current_id] = color
        return color
    map(color_node,range(len(tree_adjlist)))
    def colored_node_tostr(k,v):
        if v == 1:
            return str(k)+': red'
        elif v == 2:
            return str(k)+': blue'
        elif v == 3:
            return str(k)+': purple'
        else:
            print 'weird color value (',str(c),') here, shall not happen'
            
    color_list = sorted([colored_node_tostr(k,v) for k,v in tree_color.iteritems()])
    return '\n'.join(color_list)   
      

def longest_shared_substring(text1,text2):
    """
    Longest Shared Substring Problem: Find the longest substring shared by two strings.
    Input: Strings Text1 and Text2.
    Output: The longest substring that occurs in both Text1 and Text2.
    """
    
    def color_suffix_tree(root):
        # return a flat dictionnary where key is root's key and value is 
        # 1 for blue, 2 for red or 3=1|2 for purple
        tree_color = {}   
        s = []
        def dfs_edges(current_dict):
            if type(current_dict) != dict:
                return
            for key,child_dict in current_dict.iteritems():
                s.append((key,child_dict))
                dfs_edges(child_dict)
            return
        dfs_edges(root)
        
        # color red and blue all leaves
        keys,dicts = zip(*s)
        colors = [None]*len(dicts)
        
        for i,(k,d) in enumerate(s):
            label = text[k[0]:k[1]]
            if label[-1] == '$':
                if '#' in label:
                    colors[i] = 1
                else:
                    colors[i] = 2
                    
        print 'colors',colors
                
        def dfs_color(current_dict):
            if type(current_dict) != dict:
                return 0                
                
            i = dicts.index(current_dict)
            if colors[i] is not None:
                return colors[i]
            color = 0
            for key,child_dict in current_dict.iteritems():
                color |= dfs_color(child_dict)
            colors[i] = color
            return color

#        dfs_color(root)
        for d in dicts:
            dfs_color(d)
        print 'colors',colors
        
        print 'len(tree_color)',len(tree_color)
        
        return tree_color     

    text = text1+'#'+text2+'$'
    root = new_suffix_tree(text)
    tree_color = color_suffix_tree(root)


    print '>>> color tree---------------------------'    
    for k,v in tree_color.iteritems():
        print text[k[0]:k[1]],v
    print '<<< color tree---------------------------'    

    def dfs(current_dict,word):
        if type(current_dict) != dict:
            return
        for key, value in current_dict.iteritems():
            assert key in tree_color.keys()
#            if key not in tree_color.keys():
#                print 'key',key
#                print 'tree_color.keys()',tree_color.keys()     
            label = text[key[0]:key[1]]
            if (label == 'hello'):
                print 'first children are',value
                print 'parent color is', tree_color[key]
            candidate = word+label

            if (tree_color[key] == 3):
                print 'candidate',candidate
            
            if (tree_color[key] == 3) and (len(longests[-1]) < len(candidate)):
                # purple and longer than any longest's
                longests.append(candidate)
                # try to find deeper a longer purple one
                dfs(value,candidate)
        return
    longests = ['',]
    dfs(root,'')
    return longests[-1]
    
        



assert trie_matching('AATCGGGTTCAATCGGGGT','ATCG','GGGT') == [1,4,11,15]

assert sorted(suffix_tree_edges('anana$')) == ['$', '$', '$', '$', 'a', 'na', 'na', 'na$', 'na$']
assert sorted(new_suffix_tree_edges('anana$')) == ['$', '$', '$', '$', 'a', 'na', 'na', 'na$', 'na$']

assert longest_repeat('ATATCGTTTTATCGTT') == 'TATCGTT'

text = 'AATTTCCGACTTGCATGACGAGTCAGCGTTCCATCTGATCGAGTCTCCGAAGAACAAATACCCCTACTCAGTTGTGAGCCCCTTTACCGTGAGGACAGGGTCCTTGATGTCGTCTCCTAATTTGCGTTGCGGCTCAACATGTTGTACATAGTGGGGCCAGCCCCAGGGATTTTGTAATTTCTACACTCCATATACGGGACAAGGGTGAGCATTTCCGGGCTTGGATAGGGGCTGCAAGAAAATATCTGGACGTAAGAACTTAATGCCATTCCTACATCCTCGATACCTCGTCTGTCAGAGCAATGAGCTGGTTAGAGGACAGTATTGGTCGGTCATCCTCAGATTGGGGACACATCCGTCTCTATGTGCGTTCCGTTGCCTTGTGCTGACCTTGTCGAACGTACCCCATCTTCGAGCCGCACGCTCGACCAGCTAGGTCCCAGCAGTGGCCTGATAGAAAAATTACCTACGGGCCTCCCAATCGTCCTCCCAGGGTGTCGAACTCTCAAAATTCCCGCATGGTCGTGCTTCCGTACGAATTATGCAAACTCCAGAACCCGGATCTATTCCACGCTCAACGAGTCCTTCACGCTTGGTAGAATTTCATGCTCGTCTTTTGTATCCGTGTAAGTAGGAGGCCGCTGTACGGGTATCCCAGCCTTCGCGCTCTGCTGCAGGGACGTTAACACTCCGAACTTTCCATATACGGGACAAGGGTGAGCATTTCCGGGCTTGGATAGGGGCTGCAAGAAAATATCTGGACGTAAGAAGCTCTGAGGGATCCTCACGGAGTTAGATTTATTTTCCATATACGGGACAAGGGTGAGCATTTCCGGGCTTGGATAGGGGCTGCAAGAAAATATCTGGACGTAAGAAGAGTGATGTTTGGAATGCCAACTTCCATGCACGCCAATTGAGCAATCAGGAGAATCGAGTGCTGTTGACCTAGACCTTGTCAGAAGTATGAATTAACCGCGCGTGTAGGTTTGTCGCTCGACCTGCAAGGGTGCACAATCTGGACTGTCGTCGGCGAACGCTTTCATACGCCTACAAACCGCGTTGCTGGTCGAATCGATCTCACCACCGGCCTTGCAGGATTCTAATTATTCTCTCTCGGTGAGACTGCCGGCGGTCCATGGGTCTGTGTTTCGCTTCAAGCAGTGATATACTGGCGTTTTGTGACACATGGCCACGCACGCCTCTCGTTACTCCCAAT'
assert longest_repeat(text) == 'TTTCCATATACGGGACAAGGGTGAGCATTTCCGGGCTTGGATAGGGGCTGCAAGAAAATATCTGGACGTAAGAAG'

#assert longest_shared_substring('panama','bananas') == 'ana'
#assert longest_shared_substring('TCGGTAGATHELLOWORLDTGCGCCCACTC','AGGGGCTCGCAGTGTAHELLOWORLDAGAA') == 'HELLOWORLD'


print '\n'.join(sorted(new_suffix_tree_edges('pahelloanahelloworldma#bhelloanaworldhelloworldnas$')))
print '------------------'
print longest_shared_substring('pahelloanahelloworldma','bhelloanaworldhelloworldnas')

text = 'pahelloanahelloworldma#bhelloanaworldhelloworldnas$'

#######################

#fname = 'C:/Users/ngaude/Downloads/dataset_294_4.txt'
#with open(fname, "r") as f:
#    words = f.read().splitlines()
#t = make_trie(*words)
#s = trie_tostr(t)
#with open(fname+'.out', "w") as f:
#    f.write(s)

#fname = 'C:/Users/ngaude/Downloads/dataset_294_8.txt'
#with open(fname, "r") as f:
#    l = f.read().splitlines()
#    text = l[0]
#    words = l[1:]
#l = trie_matching(text,*words)
#print ' '.join(map(str,l))

#fname = 'C:/Users/ngaude/Downloads/dataset_296_4.txt'
#with open(fname, "r") as f:
#    l = f.read().splitlines()
#    text = l[0]
#e = suffix_tree_edges(text)
#with open(fname+'.out', "w") as f:
#    f.write('\n'.join(e))

#fname = 'C:/Users/ngaude/Downloads/dataset_296_4.txt'
#with open(fname, "r") as f:
#    l = f.read().splitlines()
#    text = l[0]
#e = new_suffix_tree_edges(text)
#with open(fname+'.out', "w") as f:
#    f.write('\n'.join(e))

#fname = 'C:/Users/ngaude/Downloads/dataset_9665_6.txt'
#with open(fname, "r") as f:
#    l = f.read().splitlines()
#    i = l.index('-')
#    edge_lines = l[:i]
#    node_lines = l[i+1:]
#s = tree_coloring(edge_lines,node_lines)
#with open(fname+'.out', "w") as f:
#    f.write(s)

#fname = 'C:/Users/ngaude/Downloads/dataset_296_6.txt'
#with open(fname, "r") as f:
#    l = f.read().splitlines()
#    s = longest_shared_substring(*l)
#with open(fname+'.out', "w") as f:
#    f.write(s)


