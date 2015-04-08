import pandas as pd
import pandas as pd


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
        if len(word) == 0:
            return
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


def longest_repeat(text,trie):
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
                label = list(text[key[0]:key[1]])
                nword = word[:] + label
                dfs(value,nword)
        return
    longests = [[],]
    if text[-1] != '$':
        text.append('$')
    dfs(trie,[])
    return longests[-1]


fpath = '/home/ngaude/workspace/data/'
fname = 'arzephir_italy_place_segment_2014-05-19.tsv'

odf = pd.read_csv(fpath+fname, sep='\t')

df = odf
df = odf[odf.aimsi % 100 == 17]


df = df[df.apply(lambda r : (r.dat_heur_debt[-8:]!=  '04:30:01') & (r.ndat_heur_debt[-8:] != '04:30:00'),axis=1)]


assert sum(map(lambda s:s[:5] == '20820',df.dat_heur_debt) ) == 0

# '2082000030F71' == '20820', '0003' , '0F71'

df = df.sort(['aimsi','dat_heur_debt'])

def segtext(df):
    df = df.sort(['aimsi','dat_heur_debt'])
    prev_aimsi = None
    prev_numr_cell = None
    text = []
    for i,r in df.iterrows():
        if (r.aimsi == prev_aimsi) and (r.nnumr_cell == prev_numr_cell):
            continue
        if (r.aimsi != prev_aimsi):
            text.append(r.aimsi)
        text.append(r.nnumr_cell[-8:])
        prev_aimsi = r.aimsi
        prev_numr_cell = r.nnumr_cell
    return text

def remove_handover(s):
    # reduce A,B,A,B in s to A3,B4 in return
    i = 0
    t = []
    while i<(len(s)-3):
        t.append(s[i])
        if (s[i] == s[i+2]) and (s[i+1]==s[i+3]):
            t.append(s[i+1])
            i +=3
        i +=1
    t += s[i:]
    return t

def pattern_find(text, pattern):
    found = []
    for i in range(len(text) - len(pattern) + 1):
        if (text[i:i+len(pattern)] == pattern):
            found.append(i)
    return found

def pattern_remove(text, pattern):
    offsets = pattern_find(text, pattern)
    s = []
    lastpos = 0
    for i in offsets:
        s += text[lastpos:i]
        lastpos = i+len(pattern)
    s += text[lastpos:]
    return s

text = segtext(df)
text = remove_handover(text)
while True:
    trie = new_suffix_tree(text)
    pattern = longest_repeat(text,trie)
    pattern_position = pattern_find(text,pattern) 
    print '|text|',len(text),'pattern',pattern,'count',len(pattern_position)
    text = pattern_remove(text,pattern)

