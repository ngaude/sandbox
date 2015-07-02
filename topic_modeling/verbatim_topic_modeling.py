# -*- coding: utf-8 -*-
"""
Created on Wed Jul 01 23:48:24 2015

@author: ngaude
"""

import time
import logging
import itertools
import pandas as pd
import numpy as np
import gensim

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO  # ipython sometimes messes up the logging setup; restore

fpath = 'E:/workspace/data/'

        
    
# df = pd.read_csv(fpath+'voy_cli_internet_verbatim.tsv',sep='\t',encoding='utf-8')
# msgs = [msg.lower() for msg in list(df.body)]
# ??? encoding char issue with french body ....:w
# CParserError: Error tokenizing data. C error: Expected 10 fields in line 4164, saw 16

import os
os.chdir('C:/Users/ngaude/Documents/GitHub/sandbox/topic_modeling')
STOPWORDS = []
with open('stop-words_french_1_fr.txt', "r") as f:
    STOPWORDS += f.read().split('\n')
with open('stop-words_french_2_fr.txt', "r") as f:
    STOPWORDS += f.read().split('\n')
STOPWORDS = set(STOPWORDS)

import string

intab = string.punctuation
outtab = ' '*len(intab)
trantab = string.maketrans(intab, outtab)

def tokenize(msg):
    msg = msg.translate(trantab).lower()
    msg.lower()
    tokens = [w for w in msg.split(' ') if (len(w)>2) and (w not in STOPWORDS) ]
    return tokens

msgs = []
with open(fpath+'voy_cli_internet_verbatim.tsv', "r") as f:
    for i,l in enumerate(f.read().split('\n')):
        columns = l.split('\t')
        if len(columns)<7:
            continue
        msg = columns[6]
        msgs.append(tokenize(msg))

words = list(itertools.chain.from_iterable(msgs))
id2word_verbatim = gensim.corpora.Dictionary(msgs)

print(id2word_verbatim)
id2word_verbatim.filter_extremes(no_below=10, no_above=0.1)

# TODO : racinisation & lemmatisation en francais 
# stemming & lemmatization

print(id2word_verbatim)

doc = "Bonjour, j'ai depuis moins d'un mois la sfr neuf box, en ce qui concerne internet ça fonctionne bien, par contre je rencontre des difficultés pour le téléphone. on a conservé l'abonnement chez france télécom donc on est en dégroupage partiel. quand je branche le téléphone directement sur la box, il ne fonctionne pas et en plus ça me déconnecte d'internet. le téléphone fonctionne à peu prèsquand il est branché sur une prise secondaire. sauriez-vous d'où vient le problème? "
bow = id2word_verbatim.doc2bow(tokenize(doc))
print(bow)

class CorpusVerbatim(object):
    def __init__(self, messages, dictionary):
        """
        Yield each document in turn, as a list of tokens (unicode strings).
        
        """
        self.messages = messages
        self.dictionary = dictionary
    
    def __iter__(self):
        for tokens in self.messages:
            yield self.dictionary.doc2bow(tokens)
    
    def __len__(self):
        return len(self.messages)

# create a stream of bag-of-words vectors
corpus_verbatim = CorpusVerbatim(msgs, id2word_verbatim)
vector = next(iter(corpus_verbatim))
print(vector)  # print the first vector in the stream

gensim.corpora.MmCorpus.serialize(fpath+'verbatim_bow.mm', corpus_verbatim)

mm_corpus = gensim.corpora.MmCorpus(fpath+'verbatim_bow.mm')
print(mm_corpus)

#### NEXT STEPS #########

# UNSUPERVISED LEARNING ....

# Latent Semantic Indexing (LSI),
# Latent Semantic Analysis (LSA), <=> ~PCA/SVD
# Latent Dirichlet Allocation (LDA), 
# Random Projections (RP) ???


#LDA

clipped_corpus = gensim.utils.ClippedCorpus(mm_corpus, 40000)  # use fewer documents during training, LDA is slow
# ClippedCorpus new in gensim 0.10.1
# copy&paste it from https://github.com/piskvorky/gensim/blob/0.10.1/gensim/utils.py#L467 if necessary (or upgrade your gensim)
lda_model = gensim.models.LdaModel(clipped_corpus, num_topics=10, id2word=id2word_verbatim, passes=4)




print(next(iter(mm_corpus)))

_ = lda_model.print_topics(-1)  # print a few most important words for each LDA topic


#### PERSIST AGAIN MODEL

lda_model.save(fpath+'lda_verbatim.model')
id2word_verbatim.save(fpath+'verbatim.dictionary')

# load the same model back; the result is equal to `lda_model`
same_lda_model = gensim.models.LdaModel.load(fpath+'lda_verbatim.model')
lda_model.print_topics(-1)

