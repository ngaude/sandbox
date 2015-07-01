# -*- coding: utf-8 -*-
"""
Created on Wed Jul 01 23:48:24 2015

@author: ngaude
"""

# http://radimrehurek.com/topic_modeling_tutorial/1%20-%20Streamed%20Corpora.html
# http://radimrehurek.com/topic_modeling_tutorial/2%20-%20Topic%20Modeling.html
# http://radimrehurek.com/topic_modeling_tutorial/3%20-%20Indexing%20and%20Retrieval.html

# import and setup modules we'll be using in this notebook
import time
import logging
import itertools

import numpy as np
import gensim

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO  # ipython sometimes messes up the logging setup; restore

fpath = 'E:/workspace/data/'
corpus = fpath+'simplewiki-20140623-pages-articles.xml.bz2'

def head(stream, n=10):
    """Convenience fnc: return the first `n` elements of the stream, as plain list."""
    return list(itertools.islice(stream, n))
    
from gensim.utils import smart_open, simple_preprocess
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki
from gensim.parsing.preprocessing import STOPWORDS

def tokenize(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]

def iter_wiki(dump_file):
    """Yield each article from the Wikipedia dump, as a `(title, tokens)` 2-tuple."""
    ignore_namespaces = 'Wikipedia Category File Portal Template MediaWiki User Help Book Draft'.split()
    for title, text, pageid in _extract_pages(smart_open(dump_file)):
        text = filter_wiki(text)
        tokens = tokenize(text)
        if len(tokens) < 50 or any(title.startswith(ns + ':') for ns in ignore_namespaces):          
            continue  # ignore short articles and various meta-articles
        yield title, tokens
        
# only use simplewiki in this tutorial (fewer documents)
# the full wiki dump is exactly the same format, but larger
stream = iter_wiki(corpus)
for title, tokens in itertools.islice(iter_wiki(corpus), 8):
    print title, tokens[:10]  # print the article title and its first ten tokens
    
id2word = {0: u'word', 2: u'profit', 300: u'another_word'}

doc_stream = (tokens for _, tokens in iter_wiki(corpus))

id2word_wiki = gensim.corpora.Dictionary(doc_stream)

print(id2word_wiki)
id2word_wiki.filter_extremes(no_below=20, no_above=0.1)

# TODO : racinisation & lemmatisation en francais 
# stemming & lemmatization
print(id2word_wiki)

doc = "A blood cell, also called a hematocyte, is a cell produced by hematopoiesis and normally found in blood."
bow = id2word_wiki.doc2bow(tokenize(doc))
print(bow)

class WikiCorpus(object):
    def __init__(self, dump_file, dictionary, clip_docs=None):
        """
        Parse the first `clip_docs` Wikipedia documents from file `dump_file`.
        Yield each document in turn, as a list of tokens (unicode strings).
        
        """
        self.dump_file = dump_file
        self.dictionary = dictionary
        self.clip_docs = clip_docs
    
    def __iter__(self):
        self.titles = []
        for title, tokens in itertools.islice(iter_wiki(self.dump_file), self.clip_docs):
            self.titles.append(title)
            yield self.dictionary.doc2bow(tokens)
    
    def __len__(self):
        return self.clip_docs

# create a stream of bag-of-words vectors
wiki_corpus = WikiCorpus(corpus, id2word_wiki)
vector = next(iter(wiki_corpus))
print(vector)  # print the first vector in the stream

gensim.corpora.MmCorpus.serialize(fpath+'wiki_bow.mm', wiki_corpus)

mm_corpus = gensim.corpora.MmCorpus(fpath+'wiki_bow.mm')
print(mm_corpus)

#### NEXT STEPS #########

# UNSUPERVISED LEARNING ....

# Latent Semantic Indexing (LSI),
# Latent Semantic Analysis (LSA), <=> ~PCA/SVD
# Latent Dirichlet Allocation (LDA), 
# Random Projections (RP) ???


#LDA

clipped_corpus = gensim.utils.ClippedCorpus(mm_corpus, 4000)  # use fewer documents during training, LDA is slow
# ClippedCorpus new in gensim 0.10.1
# copy&paste it from https://github.com/piskvorky/gensim/blob/0.10.1/gensim/utils.py#L467 if necessary (or upgrade your gensim)
lda_model = gensim.models.LdaModel(clipped_corpus, num_topics=10, id2word=id2word_wiki, passes=4)

print(next(iter(mm_corpus)))

_ = lda_model.print_topics(-1)  # print a few most important words for each LDA topic

#TFIDF

tfidf_model = gensim.models.TfidfModel(mm_corpus, id2word=id2word_wiki)
lsi_model = gensim.models.LsiModel(tfidf_model[mm_corpus], id2word=id2word_wiki, num_topics=200)

print(next(iter(lsi_model[tfidf_model[mm_corpus]])))

# We can store this "LSA via TFIDF via bag-of-words" corpus the same way again:
# cache the transformed corpora to disk, for use in later notebooks
gensim.corpora.MmCorpus.serialize(fpath+'wiki_tfidf.mm', tfidf_model[mm_corpus])
gensim.corpora.MmCorpus.serialize(fpath+'wiki_lsa.mm', lsi_model[tfidf_model[mm_corpus]])
# gensim.corpora.MmCorpus.serialize('./data/wiki_lda.mm', lda_model[mm_corpus])


## PERSISTENCY , reload pickled object

tfidf_corpus = gensim.corpora.MmCorpus(fpath+'wiki_tfidf.mm')
# `tfidf_corpus` is now exactly the same as `tfidf_model[wiki_corpus]`
print(tfidf_corpus)

lsi_corpus = gensim.corpora.MmCorpus(fpath+'wiki_lsa.mm')
# and `lsi_corpus` now equals `lsi_model[tfidf_model[wiki_corpus]]` = `lsi_model[tfidf_corpus]`
print(lsi_corpus)

##########################
# PREDICTION 
##########################


# We can use the trained models to transform new, unseen documents into the semantic space:

text = "A blood cell, also called a hematocyte, is a cell produced by hematopoiesis and normally found in blood."

# transform text into the bag-of-words space
bow_vector = id2word_wiki.doc2bow(tokenize(text))
print([(id2word_wiki[id], count) for id, count in bow_vector])

# transform into LDA space
lda_vector = lda_model[bow_vector]
print(lda_vector)
# print the document's single most prominent LDA topic
print(lda_model.print_topic(max(lda_vector, key=lambda item: item[1])[0]))

# transform into LSI space
lsi_vector = lsi_model[tfidf_model[bow_vector]]
print(lsi_vector)
# print the document's single most prominent LSI topic (not interpretable like LDA!)
print(lsi_model.print_topic(max(lsi_vector, key=lambda item: abs(item[1]))[0]))

#### PERSIST AGAIN MODEL

lda_model.save('./data/lda_wiki.model')
lsi_model.save('./data/lsi_wiki.model')
tfidf_model.save('./data/tfidf_wiki.model')
id2word_wiki.save('./data/wiki.dictionary')

# load the same model back; the result is equal to `lda_model`
same_lda_model = gensim.models.LdaModel.load('./data/lda_wiki.model')
