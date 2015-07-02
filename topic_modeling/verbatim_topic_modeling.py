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

#INFO:gensim.models.ldamodel:topic #0 (0.100): 0.029*problème + 0.026*box + 0.024*internet + 0.020*wifi + 0.020*réseau + 0.017*téléphone + 0.016*fonctionne + 0.012*connexion + 0.009*jours + 0.007*voyant
#INFO:gensim.models.ldamodel:topic #1 (0.100): 0.050*débit + 0.039*ttl + 0.037*temps + 0.024*octets + 0.022*descendant + 0.018*montant + 0.018*125 + 0.016*194 + 0.014*173 + 0.014*local
#INFO:gensim.models.ldamodel:topic #2 (0.100): 0.039*sim + 0.032*carte + 0.022*téléphone + 0.020*commande + 0.016*service + 0.014*reçu + 0.013*nouvelle + 0.012*iphone + 0.012*boutique + 0.011*demande
#INFO:gensim.models.ldamodel:topic #3 (0.100): 0.021*problème + 0.019*forum + 0.013*aide + 0.011*aider + 0.010*venir + 0.009*regrette + 0.008*questions + 0.008*service + 0.008*cas + 0.007*soirée
#INFO:gensim.models.ldamodel:topic #4 (0.100): 0.034*message + 0.027*privé + 0.023*décodeur + 0.018*prise + 0.017*box + 0.013*indésirables + 0.013*suite + 0.012*adresse + 0.011*réception + 0.010*confidentialité
#INFO:gensim.models.ldamodel:topic #5 (0.100): 0.059*numéro + 0.026*pouvez + 0.025*téléphone + 0.025*code + 0.023*mobile + 0.014*nom + 0.014*communiquer + 0.012*cordialement + 0.012*portable + 0.010*imei
#INFO:gensim.models.ldamodel:topic #6 (0.100): 0.046*mobile + 0.043*www + 0.043*bouyguestelecom + 0.031*adresse + 0.029*mail + 0.022*mms + 0.021*assistance + 0.017*lien + 0.015*envoyer + 0.014*pouvez
#INFO:gensim.models.ldamodel:topic #7 (0.100): 0.020*faut + 0.019*voir + 0.015*passe + 0.013*bonnes + 0.013*fêtes + 0.011*from + 0.011*time + 0.011*fibre + 0.011*application + 0.010*attendre
#INFO:gensim.models.ldamodel:topic #8 (0.100): 0.030*retour + 0.027*part + 0.022*cordialement + 0.022*disposition + 0.022*réponse + 0.020*reste + 0.019*besoin + 0.018*fil + 0.017*colis + 0.016*permets
#INFO:gensim.models.ldamodel:topic #9 (0.100): 0.032*mois + 0.029*offre + 0.025*facture + 0.015*you + 0.013*service + 0.013*euros + 0.012*remise + 0.010*2014 + 0.010*sensation + 0.009*mobile


#### PERSIST AGAIN MODEL

lda_model.save(fpath+'lda_verbatim.model')
id2word_verbatim.save(fpath+'verbatim.dictionary')

# load the same model back; the result is equal to `lda_model`
same_lda_model = gensim.models.LdaModel.load(fpath+'lda_verbatim.model')
lda_model.print_topics(-1)

